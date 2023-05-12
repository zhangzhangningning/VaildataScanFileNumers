import numpy as np
import argparse as args
import pandas as pd
import random
import psycopg2
import datetime
from numpy import array
import ast
import re
import math

# all_predict_cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice',
#             'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_comment',
#             'l_quantity', 'l_discount', 'l_linenumber', 'l_linestatus',
#             'l_shipmode', 'l_returnflag', 'l_tax', 'l_shipinstruct']
# all_predict_cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber','l_extendedprice','l_discount','l_tax','l_commitdate', 'l_shipinstruct','l_shipmode','l_comment']

# all_predict_cols = all_predict_cols = ['Record_Type','Registration_Class','State','County','Body_Type','Fuel_Type','Reg_Valid_Date','Color','Scofflaw_Indicator','Suspension_Indicator','Revocation_Indicator']
all_predict_cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice','l_commitdate','l_comment']
# all_predict_cols_index = [0,1,2,3,4,5,6,7,8,9,10]
# all_predict_cols = ['l_orderkey', 'l_partkey', 'l_suppkey','l_extendedprice','l_shipdate','l_comment']
all_predict_cols_index =[0,1,2,5,11,15]

# gen_random_predict_file = "/home/ning/zorder/Cardinality_Estimation_pg/gen_rand_predicts50.16.txt"
gen_random_predict_file = "/home/ning/zorder/sqls/lineitem_distinctbig_50_6.txt"
# source_file_name = "/home/ning/zorderlearn/ValidateScanFileNumbers/datasets/dmv-clean.csv"
source_file_name = "/home/ning/pg/tpch_1/lineitem.tbl"
done_reward_file = "/home/ning/zorder/New_agg_result/done_reward.txt"
selected_col_file = "/home/ning/zorder/New_agg_result/select_cols.txt"

def GenerateRandomPredicts():
    table = pd.read_csv(source_file_name,header=None,delimiter='|')
    predict = []
    for i in range(50):    
        nums_filter = random.randint(2,6)
        seed = random.randint(1,100)
        rng = np.random.RandomState(seed)
        random_row = rng.randint(0,table.shape[0])
        s = table.iloc[random_row]   
        vals = s.values
        vals = vals[all_predict_cols_index]
        idxs = rng.choice(len(all_predict_cols),replace=False,size = nums_filter)
        cols = np.take(all_predict_cols,idxs)
        ops = rng.choice(['<=','>='],size = nums_filter)
        vals = vals[idxs]
        for i in range(len(cols)):
            if str(type(vals[i])) == "<class 'str'>" :
                vals[i] = "'" + vals[i] + "'"
        if str(type(vals[i])) == "<class 'numpy.int64'>" or str(type(vals[i])) == "<class 'numpy.float64'>":
            vals[i] = str(vals[i])
        one_predict = np.array([cols,ops,vals])
        predict.append(one_predict)
        i += 1
    predict = str(predict)
    with open(gen_random_predict_file,'w') as f:
        f.write(predict)

def HandlePredictsGetArrays():
    with open(gen_random_predict_file, 'r') as f:
        input_str = f.read()
        input_list = eval(str(input_str))

    total_where_arrays = []
    for array in input_list:
        where = []
        eachsqlwhere = []
        for condition in array.T:
            column_name = condition[0]
            comparison_operator = condition[1]
            value = condition[2]
            where = np.array([column_name,comparison_operator,value])
            eachsqlwhere.append(where)
        total_where_arrays.append(eachsqlwhere)
    # print(type(total_where_arrays))
    return total_where_arrays

def GetContainANDPredicts(total_where_arrays):
    where_has_and = []
    for eachsqlwhere in total_where_arrays:
        FinalPredict = ''
        for eachpredict in eachsqlwhere:
            sqlwhere = eachpredict[0] + ' ' + eachpredict[1] + ' ' + eachpredict[2]
            FinalPredict = FinalPredict +  sqlwhere + " AND "
        FinalPredict = FinalPredict.strip(" AND ")
        where_has_and.append(FinalPredict)
    return where_has_and

def GetCompleteSql(where_has_and):
    final_workload = []
    common_sql_part = '''select * from lineitem where '''
    for each_sql_predict in where_has_and:
        if len(each_sql_predict[0]) == 0:
            final_sql = "select * from lineitem"
        else:
            final_sql = common_sql_part + each_sql_predict[0]
        final_workload.append(final_sql)
    return final_workload

def GetSQLsErows(SQLs):
    conn = psycopg2.connect(database = "postgres",user = "postgres", password = "",host = "127.0.0.1", port = "6600")
    cur = conn.cursor()
    conn.set_session(autocommit=True)

    sql = '''explain select * from lineitem where l_orderkey >= 1 '''
    cur.execute(sql)
    results=cur.fetchall()
    All_rows = GetErow(results)
    skip_files_ration = []
    read_rows = []
    
    for sql in SQLs:
        sql = "explain " + str(sql)
        cur.execute(sql)
        sql_result = cur.fetchall()
        each_sql_erow = GetErow(sql_result)
        each_sql_erow = eval(each_sql_erow)
        read_rows.append(each_sql_erow)
        skip_files_ration.append(float(each_sql_erow)/float(All_rows))
    conn.commit()
    conn.close()
    return read_rows


def GetErow(explain_result):
    pattern = r'rows=(\d+)'
    match = re.search(pattern,str(explain_result))
    if match:
        value = match.group(1)
    return value

def GetMLPredict(total_where_array,selected_cols):
    predicts = []
    for each_sql_array in total_where_array:
        predict = ''
        for each_predict in each_sql_array:
            if each_predict[0] in selected_cols:
                sorted_show = "sorted_data['" + each_predict[0] + "']" + ' ' +  each_predict[1] + ' ' + each_predict[2]
                predict = predict + '(' + sorted_show + ')' + '&'
        predict = predict.strip('&')
        if (len(predict) == 0):
            # predicts.append("sorted_data['Record_Type'] >= sorted_data['Record_Type'].min()")
            predicts.append("sorted_data['l_orderkey'] >= sorted_data['l_orderkey'].min()")
        else:
            predicts.append(predict)
    return predicts

def GetMlColumn(total_where_array,selected_cols):
    return [selected_cols] * len(total_where_array)
    columns = []
    for each_sql_array in total_where_array:
        column = []
        for each_predict in each_sql_array:
            if each_predict[0] in selected_cols:
                column.append(each_predict[0])
        if(len(column)):
            columns.append(column)
    return columns

def WriteRewards(predicte_files):
    with open(done_reward_file,'a') as f:
        final_reward = math.log(eval(str(predicte_files)))
        final_reward = -final_reward
        f.write(str(final_reward))
        f.write('\n')

def GetSelectCols():
    action_array = GetActionsArray()
    Select_Cols = GetColumnName(action_array)
    return Select_Cols

def GetActionsArray():
    with open(selected_col_file,'r') as f:
        lines = f.readlines()
        action_array = lines[-1]
    return action_array

def GetColumnName(action_array):
    column_name = []
    action_array = eval(action_array)
    for i in range(len(action_array)):
        if action_array[i] == 1:
            column_name.append(all_predict_cols[i])
    return column_name

def GetPredictsBasedSelectCols(total_where_arrays,select_cols):
    new_sqls_based_select_cols = []
    for each_sql_array in total_where_arrays:
        each_sql_all_predicts = []
        final_each_sql_predict = ''
        for each_predict in each_sql_array:
            sqlwhere = ''
            if each_predict[0] in select_cols:
                sqlwhere = (str(each_predict[0]) + ' ' + str(each_predict[1]) + ' ' +  str(each_predict[2]))
                final_each_sql_predict = final_each_sql_predict + sqlwhere + " AND "
        final_each_sql_predict = final_each_sql_predict.strip(" AND ")
        each_sql_all_predicts.append(final_each_sql_predict)
        if len(each_sql_all_predicts):
            new_sqls_based_select_cols.append(each_sql_all_predicts)
    # print(new_sqls_based_select_cols)
    return new_sqls_based_select_cols


if __name__ == "__main__":
    # GenerateRandomPredicts()
    total_where_array = HandlePredictsGetArrays()
    selected_cols = GetSelectCols()
    # selected_cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice','l_commitdate','l_comment']
    predcits_based_selected_cols = GetPredictsBasedSelectCols(total_where_array,selected_cols)
    SQLs = GetCompleteSql(predcits_based_selected_cols)
    for i in range(len(SQLs)):
        print('"' + SQLs[i] + '"')
    # predicates = GetMLPredict(total_where_array,selected_cols)
    # print(predicates)
    # columns = GetMlColumn(total_where_array,selected_cols)
    # print(columns)
    # selectivities = GetSQLsErows(SQLs)
    # print(selectivities)