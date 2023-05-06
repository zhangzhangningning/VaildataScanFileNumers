import pandas as pd
import numpy as np
import csample
import bisect
import math
from scipy.stats import rankdata
import os
import random
import itertools
from pandas.core.frame import DataFrame
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, Birch, FeatureAgglomeration, MeanShift, estimate_bandwidth
# 蓄水池采样，输入：待采样的列、采样个数
# 返回：对应的列的采样数据，作为分区边界


def ReservoirSample(data, columns, sample_nums):
    length = len(columns)
    Samples = []
    for i in range(length):
        data_col = data.loc[:, columns[i]]
        samples = csample.reservoir(data_col, sample_nums)
        samples.sort()
        Samples.append(samples)
    return Samples


# 从采样数据中获取分区边界
# INPUTS: samples 采样点、partition_nums 待分区数量、total_nums 数据总量、col_nums 列数
# OUTPUT：返回 partition_nums - 1 个边界值
def GetRangeBoundsWithSample(Samples, partition_nums, total_nums, col_nums):
    sample_nums = len(Samples[0])
    weight = total_nums / sample_nums
    target = total_nums / partition_nums

    # 不能直接 [[] * col_nums] 这是浅拷贝，其中的每个列表共用一块内存，修改其中一个另外的会一起改变
    range_bounds = [[] for i in range(col_nums)]

    for i in range(col_nums):
        step = 0
        nums = 0
        samples = Samples[i]
        values = len(set(samples))
        # 基数是否小于 partition_nums
        if values <= partition_nums:
            RangeBounds = list(set(samples))
            RangeBounds.sort()
            print(f'card less than p_nums: ')
            range_bounds[i] = RangeBounds
            # RangeBounds = RangeBounds[1:len(RangeBounds)]
        else:
            for candidate in samples:
                step += weight
                if step >= target:
                    nums += 1
                    if nums >= partition_nums:
                        break
                    step = 0
                    range_bounds[i].append(candidate)
    return range_bounds


# 排序法：排序后精确获取分区边界
# INPUTS: data 源数据、column 选作 Zorder 的列、partition_nums 分区数量
# OUTPUT: 返回 partition_nums - 1 个边界值
def SortDataAndGetRangeBound(data, columns, partition_nums):
    # order_data = rankdata(data[column], method='ordinal') - 1
    col_nums = len(columns)
    total_nums = data.shape[0]
    # 分区数量 > 数据量，则另分区数量 = 数据量
    if partition_nums > total_nums:
        partition_nums = total_nums

    range_bounds = [[] for i in range(col_nums)]

    target = total_nums // partition_nums
    for c in range(col_nums):
        RangeBounds = []
        order_data = data[columns[c]].sort_values()
        values = order_data.nunique()
        # 基数是否小于 partition_nums
        if values <= partition_nums:
            RangeBounds = data[columns[c]].value_counts().index.tolist()
            RangeBounds.sort()
        else:
            for i in range(partition_nums - 1):
                RangeBounds.append(order_data.iloc[(i+1) * target - 1])
        range_bounds[c] = RangeBounds
    return range_bounds


# INPUTS: data 原数据、RangeBounds 分区边界、Columns 需要做重排的列
def GetPartitionIDToCalDist(data, RangeBounds, Columns):
    col_nums = len(Columns)
    total_nums = data.shape[0]
    for index, row in data.iterrows():
        for c in range(col_nums):
            num = row[Columns[c]]
            id = bisect.bisect_left(RangeBounds[c], num)
            data.loc[index, Columns[c] + "_order"] = id

    return data

def GetColumn(fullname, columns):
    '''for i in range(len(columns)):
        strs = " ".join(columns[i])
        if fullname.find(strs) != -1:
            return i
    return 0'''
    pos = []
    for i in range(len(columns)):
        if fullname.find(columns[i]) != -1:
            pos.append(i)
    return pos

def GetPredicatedData(predicate, columns, columns_order):
    fullname = "/home/ning/zorderlearn/ValidateScanFileNumbers/datasets/lineitem_orderAllCols.csv"
    sorted_data = pd.read_csv(fullname, sep="|")
    predicated_data = sorted_data.loc[eval(predicate)]
    nums = predicated_data.shape[0]
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
                    new_row[c] += diff
                    new_data = pd.concat([new_data, new_row], ignore_index=True)
                    #new_data.loc[new_nums, all_columns] = row 
                    new_nums += 1
                    new_row = row
                    row[c] -= diff
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

if __name__ == "__main__":
    rootDir = '/data1/chenxu/projects/ValidateScanFileNumbers/datasets'
    seperate = '|'
    

    fullname = "/home/ning/zorderlearn/ValidateScanFileNumbers/datasets/lineitem_withorder.csv"
    """predicates = ["sorted_data['l_orderkey'] >= 3691075", "(sorted_data['l_orderkey'] >= 3691075) & (sorted_data['l_partkey'] <= 79406)", "(sorted_data['l_orderkey'] >= 3691075) & (sorted_data['l_partkey'] <= 79406) & (sorted_data['l_suppkey'] <= 4421)",
                  "(sorted_data['l_orderkey'] >= 3691075) & (sorted_data['l_partkey'] <= 79406) & (sorted_data['l_suppkey'] <= 4421) & (sorted_data['l_extendedprice'] >= 33249.6)", 
                  "(sorted_data['l_orderkey'] >= 3691075) & (sorted_data['l_partkey'] <= 79406) & (sorted_data['l_suppkey'] <= 4421) & (sorted_data['l_extendedprice'] >= 33249.6) & (sorted_data['l_shipdate'] >= '1993-05-11')", 
                  "(sorted_data['l_orderkey'] >= 3691075) & (sorted_data['l_partkey'] <= 79406) & (sorted_data['l_suppkey'] <= 4421) & (sorted_data['l_extendedprice'] >= 33249.6) & (sorted_data['l_shipdate'] >= '1993-05-11') & \
                    (sorted_data['l_comment'] >= 'ackages. furiously final pinto beans nag ')", 
                  "(sorted_data['l_orderkey'] >= 3691075) & (sorted_data['l_partkey'] <= 79406) & (sorted_data['l_suppkey'] <= 4421) & (sorted_data['l_extendedprice'] >= 33249.6) & (sorted_data['l_shipdate'] >= '1993-05-11') & \
                    (sorted_data['l_comment'] >= 'ackages. furiously final pinto beans nag ') & (sorted_data['l_receiptdate'] >= '1993-05-26')",
                  "(sorted_data['l_orderkey'] >= 3691075) & (sorted_data['l_partkey'] <= 79406) & (sorted_data['l_suppkey'] <= 4421) & (sorted_data['l_extendedprice'] >= 33249.6) & (sorted_data['l_shipdate'] >= '1993-05-11') \
            & (sorted_data['l_commitdate'] >= '1993-06-12') & (sorted_data['l_receiptdate'] >= '1993-05-26') & (sorted_data['l_comment'] >= 'ackages. furiously final pinto beans nag ')"]

    columns = [['l_orderkey'], ['l_orderkey', 'l_partkey'], ['l_orderkey', 'l_partkey', 'l_suppkey'], ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice'], 
               ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice', 'l_shipdate'], ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice', 'l_shipdate', 'l_comment'],
               ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice', 'l_shipdate', 'l_comment', 'l_receiptdate'], 
               ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_comment']]
    """
    predicates = ["sorted_data['l_shipdate'] >= '1993-05-11'", "(sorted_data['l_shipdate'] >= '1993-05-11') & (sorted_data['l_receiptdate'] >= '1993-05-11')", 
                  "(sorted_data['l_shipdate'] >= '1993-05-11') & (sorted_data['l_receiptdate'] >= '1993-05-11') & (sorted_data['l_commitdate'] >= '1993-06-12')"]
    
    columns = [['l_shipdate'], ['l_shipdate', 'l_receiptdate'], ['l_shipdate', 'l_receiptdate', 'l_commitdate']]
    # 重复 10 次采样
    original_data = pd.read_csv(fullname, sep=seperate)
    
    for t in range(1):
        seed = random.randint(0, 2023) * random.randint(0, 100)
        print(seed)
        data = original_data
        #data = original_data.sample(frac=0.01, random_state=seed)
        print(t, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        # 每一维度数据量总数
        total_nums = data.shape[0]
        print(f'total_nums: {total_nums}\n')

        # 分区数量
        partition_nums = 1000
        # 蓄水池采样点数量，从中选取分区边界
        sample_nums = partition_nums * 60

        # 待计算的列
        #col = cols
        col = ['l_linenumber', 'l_quantity', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipinstruct', 'l_shipmode']

        col_nums = len(col)
        distance = math.sqrt(col_nums) * 999

        # 蓄水池采样
        samples = ReservoirSample(data, col, sample_nums)
        print(f'Getting samples: \n')
        # 获取边界
        print(f'Getting range bounds')
        # 从采样点中获取边界
        range_bounds = GetRangeBoundsWithSample(
            samples, partition_nums, total_nums, col_nums)
        print(f'range_bounds\n')

        # 获取每个数据的分区号，并计算平均距离
        print(f'Sorting data... \n')
        # 获取分区边界并计算平均距离值
        # file_nums, count = GetPartitionIDToCalDist(data, range_bounds, col, distance)
        sorted_data = GetPartitionIDToCalDist(data, range_bounds, col)
        sorted_data.to_csv("/home/ning/zorderlearn/ValidateScanFileNumbers/datasets/lineitem_orderAllCols.csv", mode='w', sep='|')
        print("Sorting finished")
        #print(sorted_data.head())

        #Distances = [284.3265491314493, 792.0978793239323, 1077.54020607677, 1274.759977263888, 1430.2722577343843, 1554.2718279301532, 1707.320718682262, 1852.312872150045]
        """Distances = [605.2956502315003, 858.1471172743576, 1057.0848608811311]
        for i in range(len(predicates)):
            #if(i == 0):
            #    continue
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


            query = predicates[i]
            col = columns[i]

            print(col)
            col_nums = len(col)

            #Distance = math.sqrt(col_nums) * 999 / 2
            Distance = Distances[i]
            print(f'distance: {Distance}')
            #Distance = (1 + math.log(col_nums)) * 999
            
            
            
            predicated_data = sorted_data.loc[eval(query)]
            print(predicated_data.shape)
            
            sample_points = predicated_data.shape[0] if predicated_data.shape[0] < 10000 else 10000
            predicated_data = predicated_data.sample(n=sample_points) 
            
            print(f'After sampling... {predicated_data.shape}') 
            
            Orders = []
            for c in range(col_nums):
                Orders.append(col[c] + "_order")

            orders_data = predicated_data.loc[:, Orders]
            
            arr = orders_data.to_numpy()
            
            #clustering = OPTICS(min_samples=5, eps = Distance).fit(arr)
            #clustering = Birch(threshold=Distance).fit(arr)
            #clustering = FeatureAgglomeration(n_clusters=None, distance_threshold=Distance).fit(arr)
            #bandwith = estimate_bandwidth(arr, quantile=1)
            #print(bandwith)
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=Distance).fit(arr)
            #clustering = MeanShift(bandwidth=Distance).fit(arr)
            #print(clustering)
            unique, counts = np.unique(clustering.labels_, return_counts=True)

            print(f"file_id:, {unique}\n")
            print(f"numbers in this file:, {counts}\n")

            count = counts.tolist()
            print(count)
        """   
            

