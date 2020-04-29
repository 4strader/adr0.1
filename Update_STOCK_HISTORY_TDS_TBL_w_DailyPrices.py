import sqlite3
import pandas as pd
import os
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
def get_symbols(symbols_file):
    df = pd.read_csv(symbols_file, index_col=False)
    df = df.sort_values(['ticker'], ascending=True)
    return df


def exec_program(symbols_file):
    symbols_df = get_symbols(symbols_file)
    symbols_df['date'] = pd.to_datetime(symbols_df['date'], infer_datetime_format=True)
    conn = sqlite3.connect("TradeSystem.db")

    symbols_df.to_sql('STOCK_HISTORY_TDS_TBL', conn, if_exists='append', index=False)

    conn.close()

#exec_program("SO_DB_Symbols_ALL_TMP.csv", '2016-01-01', '2020-03-30', 'append')
exec_program("SHARADAR-04_28_20.csv")
# exec_program("TDV_042720.csv")

