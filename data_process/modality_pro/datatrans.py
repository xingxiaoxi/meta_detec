import pandas as pd
def xlsx_to_csv_pd():
    data_xls = pd.read_excel('../datasets/ME-selected.xlsx',index_col = 0)
    data_xls.to_csv('../datasets/feature.csv',encoding = 'utf-8')
xlsx_to_csv_pd()

