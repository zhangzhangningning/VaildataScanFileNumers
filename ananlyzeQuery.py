
if __name__ == "__main__":
    cols = [["l_orderkey <= 4340992", "l_partkey >= 50527", "l_suppkey <= 5538", "l_extendedprice >= 1477.52", "l_shipdate <= '1996-12-27'", "l_commitdate <= '1996-11-13'", "l_receiptdate <= '1997-01-11'", "l_comment >= 'ly. fluffily even dependencies wake '"\
    , "l_linenumber <= 1", "l_quantity <= 1", "l_discount <= 0.01", "l_tax >= 0.0", "l_returnflag <= 'N'", "l_linestatus >= 'O'", "l_shipinstruct <= 'TAKE BACK RETURN'", "l_shipmode <= 'REG AIR'"]]



    #cols = [('l_linenumber <= 1', 'l_quantity <= 1', 'l_discount <= 0.01', 'l_extendedprice >= 1477.52', "l_commitdate <= '1996-11-13'")]


    predicates = []
    columns = []
    
    new_cols = []
    for col in cols:
        new_col = []
        for item in col:
            new_col.extend(item.split()[:3])
        new_cols.append(new_col) 
    #print(new_cols)        
    """ for col in new_cols:
        temp = []
        temp.append(col[0])
        temp.append(col[3])
        temp.append(col[6])
        temp.append(col[9])
        columns.append(temp) """
        
    for cs in cols:
        cs = [c.split() for c in cs]

        name = []
        for c in cs:
           name.append(c[0]) 
        columns.append(name)
        # 构建布尔表达式字符串
        expr = '(' + ') & ('.join(f"sorted_data['{c[0]}'] {c[1]} {c[2]}" for c in cs) + ')' 
        predicates.append(expr) 

    print(predicates)
    print(columns)
    