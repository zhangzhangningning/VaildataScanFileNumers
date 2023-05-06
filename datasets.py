"""Dataset registrations."""
import os

import numpy as np

import common


# def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
def LoadDmv(filename='dmv-clean.csv'):
    csv_file = './datasets/{}'.format(filename)
    # cols = [
    #     'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
    #     'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
    #     'Suspension Indicator', 'Revocation Indicator'
    # ]
    cols = ['Record_Type','Registration_Class','State','County','Body_Type','Fuel_Type','Reg_Valid_Date','Color','Scofflaw_Indicator','Suspension_Indicator','Revocation_Indicator']

    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)

def LoadTPCH(filename='lineitem_withorder_sample.csv'):
    csv_file = './datasets/{}'.format(filename)
    # csv_file = os.path.join(data_dir, filename)
    cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_extendedprice', 
            'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_comment']
    cols = [i + '_order' for i in cols]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'l_shipdate': np.datetime64, 'l_commitdate': np.datetime64, 'l_receiptdate': np.datetime64}
    type_casts = {i + '_order': type_casts[i] for i in type_casts}
    # Note: Rank Space No Need type cast
    type_casts = {}
    return common.CsvTable('lineitem', csv_file, cols, type_casts, sep=',')
