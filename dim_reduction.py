"""Evaluate estimators (Naru or others) on queries."""
import argparse
import collections
import glob
import os
import pickle
import re
import time
import math
import numpy as np
import pandas as pd
import torch
import itertools
import common
import datasets
import made
import countPredictFile
import time
import sys
from sklearn.cluster import AgglomerativeClustering, estimate_bandwidth, KMeans
import get_workload_relation as wr

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device', DEVICE)

parser = argparse.ArgumentParser()

parser.add_argument('--inference-opts',
                    action='store_true',
                    help='Tracing optimization for better latency.')

parser.add_argument('--num-queries', type=int, default=20, help='# queries.')
parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
parser.add_argument('--err-csv',
                    type=str,
                    default='results.csv',
                    help='Save result csv to what path?')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
parser.add_argument('--psample',
                    type=int,
                    default=2000,
                    help='# of progressive samples to use per query.')
parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Turn on wildcard skipping.  Requires checkpoints be trained with '\
    'column masking.')
parser.add_argument('--order',
                    nargs='+',
                    type=int,
                    help='Use a specific order?')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=128,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=4, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
parser.add_argument(
    '--inv-order',
    action='store_true',
    help='Set this flag iff using MADE and specifying --order. Flag --order'\
    'lists natural indices, e.g., [0 2 1] means variable 2 appears second.'\
    'MADE, however, is implemented to take in an argument the inverse '\
    'semantics (element i indicates the position of variable i).  Transformer'\
    ' does not have this issue and thus should not have this flag on.')
parser.add_argument(
    '--input-encoding',
    type=str,
    default='binary',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='one_hot',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
    'then input encoding should be set to embed as well.')

# Transformer.
parser.add_argument(
    '--heads',
    type=int,
    default=0,
    help='Transformer: num heads.  A non-zero value turns on Transformer'\
    ' (otherwise MADE/ResMADE).'
)
parser.add_argument('--blocks',
                    type=int,
                    default=2,
                    help='Transformer: num blocks.')
parser.add_argument('--dmodel',
                    type=int,
                    default=32,
                    help='Transformer: d_model.')
parser.add_argument('--dff', type=int, default=128, help='Transformer: d_ff.')
parser.add_argument('--transformer-act',
                    type=str,
                    default='gelu',
                    help='Transformer activation.')

# Estimators to enable.
parser.add_argument('--run-sampling',
                    action='store_true',
                    help='Run a materialized sampler?')
parser.add_argument('--run-maxdiff',
                    action='store_true',
                    help='Run the MaxDiff histogram?')
parser.add_argument('--run-bn',
                    action='store_true',
                    help='Run Bayes nets? If enabled, run BN only.')

# Bayes nets.
parser.add_argument('--bn-samples',
                    type=int,
                    default=200,
                    help='# samples for each BN inference.')
parser.add_argument('--bn-root',
                    type=int,
                    default=0,
                    help='Root variable index for chow liu tree.')
# Maxdiff
parser.add_argument(
    '--maxdiff-limit',
    type=int,
    default=30000,
    help='Maximum number of partitions of the Maxdiff histogram.')

args = parser.parse_args()


def QueryToPredicate(columns, operators, vals, wrap_as_string_cols=None):
    """Converts from (c,o,v) to sql string (for Postgres)."""
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]
    v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if wrap_as_string_cols is not None:
        for i in range(len(columns)):
            if columns[i].name in wrap_as_string_cols:
                v_s[i] = "'" + str(v_s[i]) + "'"

    preds = [
        c.pg_name + ' ' + o + ' ' + v
        for c, o, v in zip(columns, operators, v_s)
    ]
    s = ' and '.join(preds)
    return ' where ' + s

def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def MakeTable():
    assert args.dataset in ['dmv-tiny', 'dmv', 'tpch']
    if args.dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif args.dataset == 'dmv':
        table = datasets.LoadDmv()
    elif args.dataset == 'tpch':
        table = datasets.LoadTPCH()

    if args.run_bn:
        return table, common.TableDataset(table)
    return table, None


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          return_col_idx=False):
    s = table.data.iloc[rng.randint(0, table.cardinality)]
    vals = s.values
    # vals = []
    # for col in all_cols:
    #     vals.append(col.data.iloc[rng.randint(0, table.cardinality)])
    # vals = np.array(vals)
    if args.dataset in ['dmv', 'dmv-tiny']:
        # TODO: Giant hack for DMV.
        vals[6] = vals[6].to_datetime64()

    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals

    vals = vals[idxs]
    if return_col_idx:
        return idxs, ops, vals

    assert len(cols) == len(ops) == len(vals)
    return cols, ops, vals


def CombinationOfPredicate(cols, ops, vals):
    """Convert a list of predicates into a single predicate with three cols."""
    assert len(cols) == len(ops) == len(vals)
    if len(cols) == 0:
        return '1=1'
    if len(cols) == 2:
        return ' AND '.join(
            [f'{c.Name()} {op} {v}' for c, op, v in zip(cols, ops, vals)])
    if len(cols) == 1:
        return f'{cols[0].Name()} {ops[0]} {vals[0]}'
    queries =  GenerateACombination(cols, ops, vals)
    return queries

    # print(QueryToPredicate(cols, ops, vals))

def generate_combinations(n):
    result = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                result.append((i,j,k))
    return result

def GenerateACombination(cols, ops, vals):
    """pick the combination from len(cols)"""
    num_filters = 3
    indexes= generate_combinations(len(cols))
    queries = []
    for idxs in indexes:
        # idxs = list(idxs)
        cols_ = np.take(cols, idxs)
        ops_ = np.take(ops, idxs)
        vals_ = np.take(vals, idxs)
        queries.append((cols_, ops_, vals_))
        # print(QueryToPredicate(cols_, ops_, vals_))
    return queries





def GenerateQuery(all_cols, rng, table, return_col_idx=False):
    """Generate a random query."""
    # num_filters = rng.randint(5, 12)
    num_filters = 5
    # all_cols = [col for col in all_cols if col.distribution_size > 50]
    print("All Columns:", all_cols)
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            return_col_idx=return_col_idx)
    return cols, ops, vals
    print(QueryToPredicate(cols, ops, vals))
    queries =  CombinationOfPredicate(cols, ops, vals)
    # print(QueryToPredicate(cols, ops, vals))
    return queries

    # return cols, ops, vals


def Query(estimators,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None):
    assert query is not None
    cols, ops, vals = query

    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # Actual.
    card = oracle_est.Query(cols, ops,
                            vals) if oracle_card is None else oracle_card
    if card == 0:
        return

    pprint('Q(', end='')
    for c, o, v in zip(cols, ops, vals):
        pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
    pprint('): ', end='')

    pprint('\n  actual {} ({:.3f}%) '.format(card,
                                             card / table.cardinality * 100),
           end='')

    for est in estimators:
        est_card = est.Query(cols, ops, vals)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
    pprint()
    return est_card, card

def ReportEsts(estimators):
    v = -1
    for est in estimators:
        print(est.name, 'max', np.max(est.errs), '99th',
              np.quantile(est.errs, 0.99), '95th', np.quantile(est.errs, 0.95),
              'median', np.quantile(est.errs, 0.5))
        v = max(v, np.max(est.errs))
    return v


def RunN(table,
         cols,
         estimators,
         rng=None,
         num=20,
         log_every=50,
         num_filters=11,
         oracle_cards=None,
         oracle_est=None):
    if rng is None:
        rng = np.random.RandomState(1234)

    last_time = None
    for i in range(num):
        do_print = False
        if i % log_every == 0:
            if last_time is not None:
                print('{:.1f} queries/sec'.format(log_every /
                                                  (time.time() - last_time)))
            do_print = True
            print('Query {}:'.format(i), end=' ')
            last_time = time.time()
        querys = GenerateQuery(cols, rng, table)
        strss = []
        ests = []
        truecards =[]
        for query in [querys]:
            col, ops, val = query
            strss.append((col[0].name, col[1].name, col[2].name))
            est, truecard = Query(estimators,
                                do_print,
                                oracle_card=oracle_cards[i]
                                if oracle_cards is not None and i < len(oracle_cards) else None,
                                query=query,
                                table=table,
                                oracle_est=oracle_est)
            ests.append(est)
            truecards.append(truecard)
        print(strss)
        print(ests)
        print(truecards)
        # assert 0

        max_err = ReportEsts(estimators)
    return False






def MakeDimReduction(scale, cols_to_train, seed, fixed_ordering=None, checkpoint=None):
    if args.inv_order:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.DimReduction(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=4,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        checkpoint=checkpoint
    ).to(DEVICE)

    return model




def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb

def GetPredicatedData(all_data, predicate, columns, columns_order):
    sorted_data = all_data
    predicated_data = sorted_data.loc[eval(predicate)]
    nums = predicated_data.shape[0]
    print("predicate data nums: ", nums)
    if nums >= 1000:
        data = predicated_data.sample(n=1000)
    else:
        col_nums = len(columns)
        all_columns = []
        all_columns += (columns + columns_order)
        print(all_columns)

        new_data = pd.DataFrame(columns=all_columns)
        target = 1000 - nums
        combinations = []
        for i in range(col_nums + 1):
            combinations += list(itertools.combinations(columns_order, i))
        diff = 0
        #print(combinations)
        new_nums = 0
        while new_nums < target:
            diff += 1
            for comb in combinations:
                for c in comb:
                    row = predicated_data.sample(n=1)
                    if new_nums > target:
                        break
                    new_row = row
                    if (new_row[c].values[0] + diff) > all_data.loc[:,c].max():
                        new_row[c] = all_data.loc[:,c].max()
                    else:    
                        new_row[c] += diff
                    new_data = pd.concat([new_data, new_row], ignore_index=True)
                    #new_data.loc[new_nums, all_columns] = row 
                    new_nums += 1
                    new_row = row
                    if (new_row[c].values[0] - diff) < all_data.loc[:,c].min():
                        new_row[c] = all_data.loc[:,c].min()
                    else:
                        new_row[c] -= diff
                    #new_data.loc[new_nums, all_columns] = row 
                    new_data = pd.concat([new_data, new_row], ignore_index=True)
                    new_nums += 1
        if new_nums > target:
            new_data = new_data.iloc[0:target,:]

        print(new_data.shape)
        data = pd.concat([predicated_data, new_data], ignore_index=True)
        #num_copies = 1000 // predicated_data.shape[0] + 1
        #new_df = pd.concat([predicated_data] * num_copies, ignore_index=True)
        #data = new_df.iloc[0:1000, :]
    #sample_nums = predicated_data.shape[0] if predicated_data.shape[0] < 1000 else 1000
    #data = predicated_data.sample(n=sample_nums)
    return data



def Main(sample_data, all_data, predicate, column, is_cal, model):
    cwd = os.getcwd()
    print(cwd)
    

    time_start = time.time()
        #sorted_data = pd.read_csv("/data1/chenxu/projects/ValidateScanFileNumbers/datasets/lineitem_orderAllCols_sample.csv", sep='|')
        #sorted_data = pd.read_csv("/home/cr/naru/datasets/lineitem_withorder_sample.csv")

    sorted_data = sample_data
    print(column)
    columns = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice',
            'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_comment',
            'l_quantity', 'l_discount', 'l_linenumber', 'l_linestatus',
            'l_shipmode', 'l_returnflag', 'l_tax', 'l_shipinstruct']
        # columns = ['Record_Type','Registration_Class','State','County','Body_Type','Fuel_Type','Reg_Valid_Date','Color','Scofflaw_Indicator','Suspension_Indicator','Revocation_Indicator']
    columns_order = [i + '_order' for i in columns]
        #predicated_data = origin_data.loc[eval(predicate)].sample() 

        #sample_nums = predicated_data.shape[0] if predicated_data.shape[0] < 10000 else 10000
        #predicated_data = predicated_data.sample(n=sample_nums)

    data = sorted_data.loc[:,columns_order]
    data = data.sample(n=10000)

    time_start = time.time()
    for c in columns:
        if c not in column:
            data[c + '_order'] = 0
        else:
            data[c + '_order'] += 1
    arr = data.to_numpy()
    print(f'points nums: {arr.shape[0]}')

    time_end = time.time()
    print("Processing data time cost: ", time_end - time_start)

    inp = torch.tensor(arr[:,:], dtype=torch.int).to(DEVICE)
    array = model(inp).cpu().detach().numpy()
    print(array.shape[0])



    if is_cal == True:
        print("To cal distance")
        return array, []        
    else:

        predicated_data = GetPredicatedData(all_data, predicate, columns, columns_order)
            # print(predicated_data)
        predicated_data = predicated_data.loc[:,columns_order]
        print(predicated_data.shape)

        for c in columns:
            if c not in column:
                predicated_data[c + '_order'] = 0
            else:
                predicated_data[c + '_order'] += 1

        parr = predicated_data.to_numpy()


        inp = torch.tensor(parr[:,:], dtype=torch.int).to(DEVICE)
            
        try:
            parray = model(inp).cpu().detach().numpy()
        except:
            exit(1)
        print(parray.shape[0])


        #用于算距离
        """ data = sorted_data.loc[:,columns_order]
        arr = data.to_numpy()
        ones = np.ones((data.shape[0], data.shape[1]))
        arr = arr + ones """



    return array, parray

def WriteRecodeSAR(selectivites_reward_cols_file,total_predict_files):
    action_array = wr.GetActionsArray()
    final_reward = math.log(eval(str(total_predict_files)))
    final_reward = -final_reward
    with open(selectivites_reward_cols_file,'a') as f:
        f.write(str(final_reward))
        f.write(' ')
        f.write(str(selectivities))
        f.write(' ')
        f.write(str(action_array))
        # f.write('\n')
def WriteDistance(distance):
    with open('/home/ning/zorderlearn/ValidateScanFileNumbers/distance.txt','w') as f:
        f.write(str(distance))
def GetDistance():
    with open('/home/ning/zorderlearn/ValidateScanFileNumbers/distance.txt','r') as f:
        distance = f.readlines()
    return distance


if __name__ == '__main__':

    #all_data = pd.read_csv("/home/ning/zorderlearn/ValidateScanFileNumbers/datasets/lineitem_orderAllCols.csv", sep='|')
    sample_data = pd.read_csv("/home/ning/zorderlearn/ValidateScanFileNumbers/datasets/lineitem_orderAllCols_sample.csv", sep='|')
    
    #all_data = pd.read_csv("/home/ning/zorderlearn/ValidateScanFileNumbers/datasets/dmv-clean_order.csv", sep=',')
    #sample_data = pd.read_csv("/home/ning/zorderlearn/ValidateScanFileNumbers/datasets/dmv-clean_order_sample.csv", sep='|')
    
    time_start = time.time()
    all_ckpts = glob.glob('./models/{}'.format(args.glob))
    # all_ckpts = glob.glob('/home/ning/zorderlearn/ValidateScanFileNumbers/models/{}'.format(args.glob))
    # all_ckpts = glob.glob('/home/ning/zorderlearn/ValidateScanFileNumbers/models/Dmv_sample-Dim4-3.2MB-model19.832-data16.399-made-resmade-hidden256_256_256_256_256-emb4-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-100epochs-seed0.pt')
    if args.blacklist:
        all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]

    selected_ckpts = all_ckpts

    print('ckpts', selected_ckpts)

    if not args.run_bn:
        # OK to load tables now
        table, train_data = MakeTable()
        cols_to_train = table.columns

    Ckpt = collections.namedtuple(
        'Ckpt', 'epoch model_bits bits_gap path loaded_model seed')
    parsed_ckpts = []

    for s in selected_ckpts:
        if args.order is None:
            z = re.match('.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt',
                         s)
        else:
            z = re.match(
                '.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+)-order.*.pt', s)
        assert z
        model_bits = float(z.group(1))
        data_bits = float(z.group(2))
        seed = int(z.group(3))
        bits_gap = model_bits - data_bits

        order = None
        if args.order is not None:
            order = list(args.order)

        if args.heads > 0:
            raise NotImplementedError
        else:
            if args.dataset in ['dmv-tiny', 'dmv', 'tpch']:
                model = MakeDimReduction(
                    scale=args.fc_hiddens,
                    cols_to_train=table.columns,
                    seed=seed,
                    fixed_ordering=order,
                    checkpoint=s)
            else:
                assert False, args.dataset

        assert order is None or len(order) == model.nin, order
        ReportModel(model)
        print('Loading ckpt:', s)
        # model.load_state_dict(torch.load(s))
        # model.eval()

        print(s, bits_gap, seed)
        # TODO: 每一列索引位置记得要加一，0代表不包含这一列

        time_end = time.time()
        print("loading time cost: ", time_end - time_start)

    # 读取第一个DataFrame对象的大小
    size1_bytes = sys.stdin.buffer.read(4)
    size1 = np.frombuffer(size1_bytes, dtype=np.uint32)[0]

    # 读取第一个DataFrame对象
    df1_bytes = sys.stdin.buffer.read(size1)
    all_data = pickle.loads(df1_bytes)

    # 读取第二个DataFrame对象的大小
    #size2_bytes = sys.stdin.buffer.read(4)
    #size2 = np.frombuffer(size2_bytes, dtype=np.uint32)[0]

    # 读取第二个DataFrame对象
    #df2_bytes = sys.stdin.buffer.read(size2)
    #sample_data = pickle.loads(df2_bytes)

    selectivites_reward_cols_file = '/home/ning/zorder/New_agg_result/selectivites_reward_cols.txt'

    """sample_cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice',
            'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_comment',
            'l_quantity', 'l_discount', 'l_linenumber', 'l_linestatus',
            'l_shipmode', 'l_returnflag', 'l_tax', 'l_shipinstruct']"""
    
    # 用于算聚类的距离参数，选不同的列需要更改
    #sample_cols =['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice','l_shipdate', 'l_comment', 'l_discount', 'l_linenumber','l_shipmode', 'l_tax', 'l_shipinstruct']
    #sample_cols =['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice','l_commitdate', 'l_comment', 'l_discount', 'l_linenumber','l_shipmode', 'l_tax', 'l_shipinstruct']
    # sample_cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber','l_extendedprice','l_discount','l_tax','l_commitdate', 'l_shipinstruct','l_shipmode','l_comment']
    sample_cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice','l_commitdate','l_comment']
    # sample_cols = ['Record_Type','Registration_Class','State','County','Body_Type','Fuel_Type','Reg_Valid_Date','Color','Scofflaw_Indicator','Suspension_Indicator','Revocation_Indicator']


    sample_cols_order = [i + '_order' for i in sample_cols]
    all_cols = sample_cols + sample_cols_order
    print(sample_cols)



    total_where_array = wr.HandlePredictsGetArrays()
    selected_cols = wr.GetSelectCols()
    predicates = wr.GetMLPredict(total_where_array,selected_cols)
    # print(predicates)
    predcits_based_selected_cols = wr.GetPredictsBasedSelectCols(total_where_array,selected_cols)
    columns = wr.GetMlColumn(total_where_array,selected_cols)
    SQLs = wr.GetCompleteSql(predcits_based_selected_cols)
    selectivities = wr.GetSQLsErows(SQLs)
    print(selectivities)

    distance = GetDistance()
    if len(distance) == 0:
        arr, parr = Main(sample_data, all_data, "null", sample_cols, True, model)
        bandwidth = estimate_bandwidth(arr, quantile=1)
        WriteDistance(bandwidth)
    else:
        bandwidth = eval(distance[0])
    print("bandwidth: ", bandwidth)

    res = []
    #for i in range(1):
    total_predicts_files = 0
    full_scan_files = 0
    full_scan_cnts = 0
    for i in range(len(predicates)):
        if predicates[i] == "sorted_data['l_orderkey'] >= sorted_data['l_orderkey'].min()" :
            full_scan_cnts += 1
            print("need to full scan")
            if full_scan_cnts > 1:
                total_predicts_files += full_scan_files
                res.append(full_scan_files) 
                print("full scan uses last result")
                continue
        arr, parr = Main(sample_data, all_data, predicates[i], columns[i], False, model)
        predict_nums = parr.shape[0]
        print(f'predict nums: {predict_nums}')
        array = np.vstack((parr, arr))
        #print(array.shape)
        #bandwith = estimate_bandwidth(array, quantile=1)
        #print(bandwith)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(columns[i])
        #clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.3276936978336165).fit(arr)
        time_start = time.time()
        #labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1.1486221077273846).fit_predict(array)
        time_end = time.time()
        print("cluster time", time_end - time_start)
        #print(labels.shape)
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=bandwidth).fit_predict(array)
        p_labels = labels[0:predict_nums]
        #print(p_labels.shape)
        unique, counts = np.unique(labels, return_counts=True)

        print(f"All file_id:, {unique}\n")
        print(f"numbers in each files:, {counts}\n")


        """ # 如果聚簇数量过少，需要重新聚簇
        if len(unique) < 5:
            new_labels = KMeans(n_clusters=2).fit(array).labels_
        unique, counts = np.unique(new_labels, return_counts=True)

        print(f"New file_id:, {unique}\n")
        print(f"numbers in each files:, {counts}\n")"""


        unique, counts = np.unique(p_labels, return_counts=True)
        print(f"Scan file_id:, {unique}\n")
        print(f"numbers in each file:, {counts}\n")

        nums = len(unique)
        # 如果将 1000 个点进行聚簇时，聚簇的数量太少，需要进行切分 
        #for c in counts:
        #    nums += c // 60 

        print(selectivities[i], predict_nums, nums)

        predict_nums = selectivities[i] if selectivities[i] < 1000 else predict_nums
        files = selectivities[i] / predict_nums * nums
        #files = selectivities[i] / 1000 * nums
        total_predicts_files += files
        print(files)
        if full_scan_cnts == 1:
            full_scan_files = files
        res.append(files)
        #print(files)
        #count = counts.tolist()
        #print(count)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end")
    print(res)
    print(predicates)
    wr.WriteRewards(total_predicts_files)
    WriteRecodeSAR(selectivites_reward_cols_file,total_predicts_files)