# %%
import pandas as pd
import datetime as dt
import numpy as np
import sys

from datetime import timedelta

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)

dates_df = pd.DataFrame()
symbols_df = pd.DataFrame()

def day_chgperc(row):

    row['ocpercchgL'] = row['C-O%']
    row['ocpercchgS'] = -1* row['C-O%']
    row['LongsExit'] = row['Open'] - (20 * row['ADR'])
    row['ShortsExit'] = row['Open'] + (20 * row['ADR'])
    row['SLocpercchgL'] = ((row['LongsExit']-row['Open'])/row['Open'])*100
    row['SLocpercchgS'] = ((row['Open']-row['ShortsExit'])/row['Open'])*100
    row['LmaxdayPL'] = np.maximum(row['ocpercchgL'],row['SLocpercchgL']  )
    row['SmaxdayPL'] = np.maximum(row['ocpercchgS'], row['SLocpercchgS'])

    LongsFillCond = ((row['Side'] == 'L') & (row['Open'] <= row['LOrderPr']))
    ShortsFillCond = ((row['Side'] == 'S') & (row['Open'] >= row['SOrderPr']))

    row['percchg'] = np.where(LongsFillCond, row['LmaxdayPL'], (np.where(ShortsFillCond, row['SmaxdayPL'], 99 )) )

    return row


def prep_bt_data(input_file, hist_st_date):
    df = pd.read_csv(input_file, index_col=False)
    df.drop(['Adj Close', 'Volume'], 1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df = df.sort_values(['Symbol', 'Date'], ascending=True)
    #df = df[(df['Date'] >= hist_st_date) & (df['Close'] >= 10) & (df['Close'] <= 300) ]
    df = df[(df['Date'] >= hist_st_date)]
    df = df.round(decimals=2)

    return df


def calc_winperc(winct,totalct):

    if not totalct:
        winperc = 0
    else:
        winperc = (winct / totalct) * 100
    #return int(round(winperc))
    return winperc


def get_spy_data(input_file, hist_st_date):
    df = pd.read_csv(input_file, index_col=False)
    df.drop(['Adj Close', 'Volume'], 1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df = df[(df['Date'] > hist_st_date)]

    return df

def build_symbol_data(input_df, adrcutoffdate):

    symbols_df['Symbol'] = input_df['Symbol'].unique()

    output_list = []
    for row in symbols_df.itertuples():
        output_iterdf = input_df.loc[input_df['Symbol'] == row.Symbol]
        output_iterdf.sort_values(['Date', 'Symbol'], inplace = True, ascending=True)
        output_iterdf['PrevClose'] = output_iterdf['Close'].shift(1)
        #output_iterdf['pct_chg'] = output_iterdf['Close'].pct_change()
        #output_iterdf['SPY_pct_chg'] = output_iterdf['SPYCLOSE'].pct_change()
        #output_iterdf['beta'] = output_iterdf.rolling(10).cov().unstack()['pct_chg']['SPY_pct_chg']
        #output_iterdf['beta'] = ((output_iterdf[['SPY_pct_chg']].rolling(window = 252).cov(other = output_iterdf['pct_chg'].rolling(window=252)))['SPY_pct_chg']/(output_iterdf['SPY_pct_chg'].rolling(window =252).var())).dropna()
        output_iterdf['O-PC%'] = ((output_iterdf['Open']-output_iterdf['PrevClose'])/output_iterdf['PrevClose'])*100
        output_iterdf['C-O%'] = ((output_iterdf['Close']-output_iterdf['Open'])/output_iterdf['Open'])*100
        output_iterdf['ADR'] = (output_iterdf['High']-output_iterdf['Low']).rolling(window=20).mean()
        output_iterdf['52H'] = (output_iterdf['High']).rolling(window=252).max()
        output_iterdf['dist_52H'] = (((output_iterdf['52H'] - output_iterdf['Close']) / output_iterdf['Close']) * 100).round(decimals=2)
        #output_iterdf['SMA'] = (output_iterdf['Close']).rolling(window=adrrange).mean()
        output_iterdf = output_iterdf.iloc[255:]
        output_list.append(output_iterdf)
    output_df = pd.concat(output_list)
    output_df = output_df.dropna(how='any')
    output_df = output_df[(output_df['Date'] > adrcutoffdate)]

    return output_df

def exec_test(input_file, hist_st_date, adrcutoffdate, outputfile, l_adj, s_adj):

    print('Start Time:', dt.datetime.now())

    hist_data_df = prep_bt_data(input_file, hist_st_date)

    hist_data_df2 = build_symbol_data(hist_data_df, adrcutoffdate)
    hist_spy_df = get_spy_data('SPY_Daily_010110_080919.csv', hist_st_date)

    dates_df['Date'] = hist_data_df2['Date'].unique()
    dates_df.sort_values(['Date'], inplace = True, ascending= True)
    dates_df.drop(dates_df.index[0], inplace= True)

    hist_list = []
    summary_list = []

    for row_date in dates_df.itertuples():
        print(row_date.Date)
        hist_perdate_df = hist_data_df2[(hist_data_df2['Date'] == row_date.Date)]
        #hist_perdate_df = hist_perdate_df.sort_values(['Symbol'], ascending=True)
        hist_perdate_df = hist_perdate_df.sort_values(['dist_52H'], ascending=True)
        MedianSymbolL = hist_perdate_df.dist_52H.nsmallest(200).iloc[-1]
        MedianSymbolS = hist_perdate_df.dist_52H.nlargest(200).iloc[-1]
        io = float(hist_spy_df.loc[(hist_spy_df['Date'] == row_date.Date), 'IO'])
        disc_orig = float(hist_spy_df.loc[(hist_spy_df['Date'] == row_date.Date), 'DISC'])
        spyocperc = float(hist_spy_df.loc[(hist_spy_df['Date'] == row_date.Date), 'OC'])
        hist_perdate_df['spyocperc'] = spyocperc
        adjdisc_l = disc_orig - l_adj
        adjdisc_s = disc_orig - s_adj
        lorderadj = float(1 + ((io-adjdisc_l)/100))
        sorderadj = float(1 + ((io+adjdisc_s)/100))
        hist_perdate_df['lorderadj'] = lorderadj
        hist_perdate_df['sorderadj'] = sorderadj
        hist_perdate_df['LOrderPr'] = lorderadj*hist_perdate_df['PrevClose']
        hist_perdate_df['SOrderPr'] = sorderadj*hist_perdate_df['PrevClose']
        #hist_perdate_df['LOrderPr'] = (1 + (((io*hist_perdate_df['BETA'])-adjdisc_l)/100)) * hist_perdate_df['PrevClose']
        #hist_perdate_df['SOrderPr'] = (1 + (((io*hist_perdate_df['BETA'])+adjdisc_s)/100)) * hist_perdate_df['PrevClose']
        #hist_perdate_df['BetaAdjOpen'] = (1 + ((io * hist_perdate_df['BETA']) / 100)) * hist_perdate_df['PrevClose']
        #hist_perdate_df['BetaOpenPC%chg'] = ((hist_perdate_df['BetaAdjOpen']-hist_perdate_df['PrevClose'])/hist_perdate_df['PrevClose'])*100

        hist_perdate_df['IO'] = io
        hist_perdate_df['adjdisc_l'] = adjdisc_l
        hist_perdate_df['adjdisc_s'] = adjdisc_s

        LongsCond = (hist_perdate_df['dist_52H'] <= MedianSymbolL)
        ShortsCond = (hist_perdate_df['dist_52H'] >= MedianSymbolS)

        hist_perdate_df['Side'] = np.where(LongsCond, 'L', (np.where(ShortsCond, 'S', 'N')))
        hist_perdate_df['percchg'] = day_chgperc(hist_perdate_df)['percchg']

        hist_list.append(hist_perdate_df)

        summary_df = {'Date': row_date.Date}
        summary_df['TotalCt'] = hist_perdate_df.shape[0]
        fillct_cond = (hist_perdate_df['percchg'] < 99)
        profit_cond = ((hist_perdate_df['percchg'] < 99) & (hist_perdate_df['percchg'] > 0))
        fillct_condL1 = (hist_perdate_df['Open'] <= hist_perdate_df['LOrderPr'])
        fillct_condS1 = (hist_perdate_df['Open'] >= hist_perdate_df['SOrderPr'])
        fillct_condL2 = (hist_perdate_df['Side'] == 'L')
        fillct_condS2 = (hist_perdate_df['Side'] == 'S')
        fillct_condL = (fillct_condL2 & fillct_condL1)
        fillct_condS = (fillct_condS2 & fillct_condS1)
        summary_df['avg_chgperc'] = hist_perdate_df.loc[fillct_cond, 'percchg'].mean()
        summary_df['FillsCt'] = fillct_cond.sum()
        summary_df['FillPerc'] = calc_winperc(summary_df['FillsCt'], (fillct_condL2.sum()+fillct_condS2.sum()))
        summary_df['TotalFillsWinCt'] = ((hist_perdate_df['percchg'] < 99) & (hist_perdate_df['percchg'] > 0)).sum()
        summary_df['TotalFillsWinPerc'] = calc_winperc(summary_df['TotalFillsWinCt'], summary_df['FillsCt'])
        summary_df['LongsCt'] = fillct_condL2.sum()
        summary_df['LongsFillsCt'] = fillct_condL.sum()
        summary_df['LongsFillPerc'] = calc_winperc(summary_df['LongsFillsCt'], summary_df['LongsCt'])
        summary_df['LongsWinCt'] = (fillct_condL & profit_cond).sum()
        summary_df['LongsWinPerc'] = calc_winperc(summary_df['LongsWinCt'], summary_df['LongsFillsCt'])
        summary_df['Lavg_chgperc'] = hist_perdate_df.loc[fillct_condL, 'percchg'].mean()
        summary_df['ShortsCt'] = fillct_condS2.sum()
        summary_df['ShortsFillsCt'] = fillct_condS.sum()
        summary_df['ShortsFillPerc'] = calc_winperc(summary_df['ShortsFillsCt'], summary_df['ShortsCt'])
        summary_df['ShortsWinCt'] = (fillct_condS & profit_cond).sum()
        summary_df['ShortsWinPerc'] = calc_winperc(summary_df['ShortsWinCt'], summary_df['ShortsFillsCt'])
        summary_df['Savg_chgperc'] = hist_perdate_df.loc[fillct_condS, 'percchg'].mean()
        summary_df['lorderadj'] = lorderadj
        summary_df['sorderadj'] = sorderadj
        summary_df['IO'] = io
        summary_df['Discount'] = disc_orig
        summary_df['adjdisc_l'] = adjdisc_l
        summary_df['adjdisc_s'] = adjdisc_s

        summary_df['SPYOCPerc'] = spyocperc
        summary_df['stocks_io%'] = hist_perdate_df['O-PC%'].mean()
        #summary_df['stocks_ioB%'] = hist_perdate_df['BetaOpenPC%chg'].mean()
        summary_list.append(summary_df)
    results_df = pd.DataFrame(summary_list,
                              columns=['Date', 'avg_chgperc', 'TotalCt', 'FillsCt', 'FillPerc', 'TotalFillsWinCt',
                                       'TotalFillsWinPerc', 'LongsCt', 'LongsFillsCt', 'LongsFillPerc', 'LongsWinCt',
                                       'LongsWinPerc', 'Lavg_chgperc',
                                       'ShortsCt', 'ShortsFillsCt', 'ShortsFillPerc', 'ShortsWinCt', 'ShortsWinPerc',
                                       'Savg_chgperc', 'IO', 'Discount', 'adjdisc_l','adjdisc_s','lorderadj', 'sorderadj', 'SPYOCPerc','stocks_io%', 'stocks_ioB%'])
    output1_df = pd.concat(hist_list)
    #output1_df.to_csv("iodsp500_52H200_2013_080919_25_1dtl.csv")

    print('Total Avg:',results_df['avg_chgperc'].mean())
    print('Lavg:',results_df['Lavg_chgperc'].mean())
    print('Savg:',results_df['Savg_chgperc'].mean())
    print('FillPerc:', results_df['FillPerc'].mean())
    print('LongsFillPerc:', results_df['LongsFillPerc'].mean())
    print('ShortsFillPerc:', results_df['ShortsFillPerc'].mean())
    results_df.to_csv(outputfile)


#exec_test('sp500_2012_to_080919.csv', '2012-01-01', '2011-02-05', 'test1.csv')
exec_test('sp500_010112_080919.csv', '2012-01-01', '2013-01-05', 'iodsp400_52H150_2013_080819_35_2.csv', float(.35), float(.2))




# %%

# %%
