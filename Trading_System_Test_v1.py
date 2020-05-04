import datetime as dt
import math
import sqlite3
import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import monthly_returns_heatmap as mrh
register_matplotlib_converters()

from calendar import isleap

desired_width = 360
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)
dates_df = pd.DataFrame()

def get_sqlcolumns_list():
    list = [
        'date',  # common
        'ticker',  # common
        # 'open',
        # 'high',
        # 'low',
        'close',  # common
        # 'volume',
        'w_streak',  # S1
        'l_streak',  # S2
        'nextOpen',  # common
        # 'nextClose',
        # 'next_OPCPerc',
        'next_COPerc',  # common
        # 'next_CCPerc',
        # 'avgvol10',
        # 'avgvol50',
        'atr5',  # common
        'natr5',  # common
        # 'natr10',
        'natr50',
        'roc3',  # S6
        'roc6',  # L1,#S2
        # 'roc10',
        # 'roc20',
        'roc50',
        'roc200',
        'rsi3',  # S1,#L1
        # 'rsi10',
        'rsi4',  # L2
        'rsi14',  # L3
        'sma5',  # S6
        # 'sma7',
        # 'sma10',
        # 'sma14',
        # 'sma25',
        # 'sma50',
        'sma100',  # common
        # 'sma150',
        # 'sma200',
        # 'ema5',
        # 'ema7',
        # 'ema10',
        # 'ema14',
        # 'ema25',
        # 'ema50',
        'ema100',#L1
        # 'ema150',
        # 'ema200',
        # 'adx5',
        'adx7',  # s1,L1
        'adx14',#L4,S4
        'percBB5',  # L3
        'percBB14',#L4,S4
        # 'max5',
        # 'min5',
        # 'max14',
        # 'min14',
        # 'max50',
        # 'min50',
        # 'max200',
        # 'min200',
        # 'max252',
        # 'min252',
        'SPYClose',  # common
        # 'SPYNext_OPCPerc',#common
        'SPYNext_COPerc',  # common
        'SPYNext_CCPerc',  # common
        'FVNextOpen',  # common
        # 'SPYsma5',
        # 'SPYsma7',
        # 'SPYsma10',
        # 'SPYsma14',
        # 'SPYsma25',
        # 'SPYsma50',
        # 'SPYsma100',
        # 'SPYsma200',
        # 'SPYema5',
        # 'SPYema7',
        # 'SPYema10',
        # 'SPYema14',
        # 'SPYema25',
        # 'SPYema50',
        'SPYema100',  # S6
        # 'SPYema200',
        # 'shortsOrderPrice',#common
        # 'longsOrderPrice'#common
    ]
    return list

def get_FV_OpenStats(df, hist_st_date, hist_end_date):

    uniqueDates_df = pd.DataFrame()
    uniqueDates_df['date'] = df['date'].unique()
    list = []
    for rowDate in uniqueDates_df.itertuples():
        rowDate_df = df[df['date'] == rowDate.date]
        totalSymbols = rowDate_df.shape[0]
        symbolsAboveFV = rowDate_df[rowDate_df['nextOpen'] > rowDate_df['FVNextOpen']].shape[0]
        winSymbolsAboveFV = rowDate_df[(rowDate_df['nextOpen'] > rowDate_df['FVNextOpen']) & (rowDate_df['next_COPerc'] > 0)].shape[0]

        symbolsBelowFV = rowDate_df[rowDate_df['nextOpen'] < rowDate_df['FVNextOpen']].shape[0]
        # SPYNext_OPCPerc = rowDate_df['SPYNext_OPCPerc'].mean()
        # SPYNext_COPerc = rowDate_df['SPYNext_COPerc'].mean()
        # SPYNext_CCPerc = rowDate_df['SPYNext_CCPerc'].mean()

        winSymbolsBelowFV = rowDate_df[(rowDate_df['nextOpen'] < rowDate_df['FVNextOpen'])& (rowDate_df['next_COPerc'] > 0)].shape[0]
        try:
            aboveFVWinPerc = winSymbolsAboveFV / symbolsAboveFV * 100
            belowFVWinPerc = winSymbolsBelowFV / symbolsBelowFV * 100
        except ZeroDivisionError:
            aboveFVWinPerc = 0
            belowFVWinPerc = 0
        if (aboveFVWinPerc>belowFVWinPerc):
            winDayType = 'aboveFV'
        else:
            winDayType = 'belowFV'

        data = {'date':rowDate.date,'winDayType':winDayType, 'totalSymbols':totalSymbols, 'symbolsAboveFV':symbolsAboveFV, 'winSymbolsAboveFV':winSymbolsAboveFV,'aboveFVWinPerc':aboveFVWinPerc,
                'symbolsBelowFV':symbolsBelowFV,'winSymbolsBelowFV':winSymbolsBelowFV, 'belowFVWinPerc':belowFVWinPerc}
                # 'SPYNext_OPCPerc':SPYNext_OPCPerc , 'SPYNext_COPerc':SPYNext_COPerc, 'SPYNext_CCPerc':SPYNext_CCPerc }

        list.append(data)
    output_df = pd.DataFrame(list)
    output_df = output_df.round(decimals=2)
    return output_df

def chunks_to_df(gen):
    chunks = []
    for df in gen:
        chunks.append(df)
    return pd.concat(chunks).reset_index().drop('index', axis=1)


def get_stockhistory(col_list, hist_st_date, hist_end_date, source):
    # conn = sqlite3.connect("NewDatabase.db")
    conn = sqlite3.connect("TradeSystem.db")
    conn.text_factory = lambda b: b.decode(errors='ignore')
    start = timer()

    select_string = ','.join(col_list)
    sql_dynamic_params = dict(columnStr = select_string)

    if source == 'YAHOO':
        # sql = '''select {columnStr} from Technical_Indicators_Yahoo_Master418 where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)
        # sql = '''select {columnStr} from Technical_Indicators_Yahoo_Master_TBL where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)
        # sql = '''select {columnStr} from Technical_Indicators_Yahoo_Master where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)
        sql = '''select {columnStr} from Technical_Indicators_Yahoo_Helper where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)
    elif source == 'SHARADAR':
        # sql = '''select {columnStr} from Technical_Indicators_Master421 where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)
        # sql = '''select {columnStr} from Technical_Indicators_Master418 where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)
        # sql = '''select {columnStr} from Technical_Indicators_SHRDR_Master where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)
        # sql = '''select {columnStr} from Technical_Indicators_SHRDR_Master_TBL where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)
        sql = '''select {columnStr} from Technical_Indicators_Master where date >= (?) and date <= (?)'''.format(**sql_dynamic_params)

    df_chunks = pd.read_sql_query(sql, conn, params=(hist_st_date, hist_end_date), chunksize=1000000)

    # df_chunks = pd.read_sql_query(
    #     'select * from Technical_Indicators_Master where date >= (?) and date <= (?)', conn,
    #     params=(hist_st_date, hist_end_date), chunksize=1000000)
    df = chunks_to_df(df_chunks)

    # spy_df = pd.read_sql('select * from spyFor_TradingSystem where date >= (?) and date <= (?)', conn,
    spy_df = pd.read_sql('select * from SPY_For_TDS_v1 where date >= (?) and date <= (?)', conn,
                         params=(hist_st_date, hist_end_date))
    conn.close()
    print('DB-records', df.shape[0])
    print('DB query time for Technical_Indicators_Master', timer() - start)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    spy_df['date'] = pd.to_datetime(spy_df['date'], infer_datetime_format=True)

    return df, spy_df


def streaks(df):
    # df['chg4Streak'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
    sign = np.sign(df['chg4Streak'])
    s = sign.groupby((sign != sign.shift()).cumsum()).cumsum()
    return df.assign(w_streak=s.where(s > 0, 0.0), l_streak=s.where(s < 0, 0.0).abs())

def plot(df):
    #fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(df['date'], df['equity'])
    # ax1.set_ylabel('Equity')
    #
    # ax2 = ax1.twinx()
    # ax2.plot(df['date'], df['drawDowns'], 'r-')
    # ax2.set_ylabel('drawDowns', color='r')
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('r')

    # df.plot(x='date', y='drawDowns')
    mrh.plot(df, eoy=True, figsize=(14,9), cbar=False, square=True)


    plt.show()

def calcCAGR(df):
    end_date = df['date'].max()
    start_date = df['date'].min()
    end_price = df.loc[(df['date'] == end_date), 'cumReturns'].values[0]
    start_price = df.loc[df['date'] == start_date, 'cumReturns'].values[0]
    diffyears = end_date.year - start_date.year
    difference = end_date - start_date.replace(end_date.year)
    days_in_year = isleap(end_date.year) and 366 or 365
    number_years = diffyears + difference.days / days_in_year
    cagr = (pow((end_price / start_price), (1 / number_years)) - 1) * 100
    total_return = round(((end_price - start_price) / start_price) * 100)
    return total_return, cagr

def get_SPYCC_BenchMark_Metrics(capital, daily_metrics_dfSPYCO, spy_df):
    spy_df['startingCapital'] = capital
    spy_df['dayPL'] = (spy_df['SPYNext_CCPerc'] * capital) / 100
    spy_df['longsCapital'] = capital
    spy_df['shortsCapital'] = 0
    spy_df['symbolsCount'] = 1
    spy_df['totalShares'] = 1
    spy_df['totScreenedSymbols'] = 1
    spy_df['totOrdersSubmitted'] = 1
    spy_df['totalFilledOrders'] = 1

    daily_metrics_df_SPYCCAllDays = build_daily_metrics(spy_df)
    finalPerf_dfSPYCCAllDays = get_final_perf(daily_metrics_df_SPYCCAllDays, daily_metrics_dfSPYCO)
    print("SPY BENCHMARK METRICS CC", '\n', finalPerf_dfSPYCCAllDays.to_string(index=False))

def write_results_to_db(result_bydate_df, spy_df, summary_df):
    conn = sqlite3.connect("TradeSystem.db")
    summary_df.to_sql('AllSymbolsResults_TBL', conn, if_exists='replace', index=False)
    result_bydate_df.sort_values(['date'], inplace=True, ascending=False)
    result_bydate_df.to_sql('ByDateResults_TBL', conn, if_exists='replace', index=False)
    spy_df.to_sql('SPYResults_TBL', conn, if_exists='replace', index=False)
    conn.close()

def build_daily_metrics(result_bydate_df, hedge='N'):
    result_bydate_df['spyHedgeCapital'] = result_bydate_df['shortsCapital'] - result_bydate_df['longsCapital']
    result_bydate_df['hedgeCapPerc'] = 0
    if hedge == 'Y':
        # result_bydate_df['spyHedgeCapital'] = result_bydate_df['shortsCapital'] - result_bydate_df['longsCapital']

        # result_bydate_df['spyHedgeCapital'] = np.where((result_bydate_df['spyHedgeCapital'].abs()/(result_bydate_df['shortsCapital'] + result_bydate_df['longsCapital']))*100> 20,
        #                                      result_bydate_df['spyHedgeCapital']*1.2,result_bydate_df['spyHedgeCapital'])

        result_bydate_df['spyHedgePL'] = (result_bydate_df['spyHedgeCapital'] * result_bydate_df[
            'tradeSideNextSPYCOPerc']) / 100
        result_bydate_df['startingCapital'] = result_bydate_df['startingCapital'] + result_bydate_df[
            'spyHedgeCapital'].abs()
        result_bydate_df['dayPL'] = result_bydate_df['dayPL'] + result_bydate_df['spyHedgePL']
        result_bydate_df['hedgeCapPerc'] = (result_bydate_df['spyHedgeCapital'].abs() / result_bydate_df[
            'startingCapital']) * 100


    result_bydate_df['endingCapital'] = result_bydate_df['startingCapital'] + result_bydate_df['dayPL']
    result_bydate_df['equity'] = result_bydate_df['dayPL'].cumsum()
    result_bydate_df['returnPerc'] = (result_bydate_df['dayPL'] / result_bydate_df['startingCapital']) * 100
    result_bydate_df['cumReturns'] = ((result_bydate_df['returnPerc'] / 100) + 1).cumprod()
    result_bydate_df['drawDowns'] = 1 - result_bydate_df['cumReturns'].div(result_bydate_df['cumReturns'].cummax())
    result_bydate_df['drawDowns'] = result_bydate_df['drawDowns'] * 100
    """reusing streaks method to calculate streak of win and losing days"""
    result_bydate_df['chg4Streak'] = np.where((result_bydate_df['returnPerc'] > 0), 1, -1)
    result_bydate_df = streaks(result_bydate_df)
    result_bydate_df['win_streak'] = result_bydate_df['w_streak']
    result_bydate_df['loss_streak'] = result_bydate_df['l_streak']
    """Calculating max Drawdown duration and max duration without a Drawdown by reusing streaks"""
    result_bydate_df['chg4Streak'] = np.where(result_bydate_df['drawDowns'] > 0, 1, -1)
    result_bydate_df = streaks(result_bydate_df)
    result_bydate_df['DDw_streak'] = result_bydate_df['w_streak']
    result_bydate_df['DDl_streak'] = result_bydate_df['l_streak']
    # result_bydate_df['totScreenedSymbols'] = result_bydate_df['totScreenedSymbols'].max()
    # result_bydate_df['totOrdersSubmitted'] = result_bydate_df['totOrdersSubmitted'].max()
    # result_bydate_df['totalFilledOrders'] = result_bydate_df['totalFilledOrders'].max()
    return result_bydate_df

def get_final_perf(result_bydate_df, result_bydate_dfSPYCO):
    '''losses and streaks'''
    maxLoss = result_bydate_df['dayPL'].min()
    maxProfit = result_bydate_df['dayPL'].max()
    maxWStreak = result_bydate_df['win_streak'].max()
    maxLStreak = result_bydate_df['loss_streak'].max()
    maxLStreakDate = result_bydate_df.loc[result_bydate_df['loss_streak'] == maxLStreak, 'date'].values[0]
    avgPLWindays = result_bydate_df[result_bydate_df['dayPL'] > 0]['dayPL'].mean()
    avgPLLossdays = result_bydate_df[result_bydate_df['dayPL'] < 0]['dayPL'].mean()
    winByLossPL = abs(round(avgPLWindays / avgPLLossdays, 2))
    avgretPcWin = result_bydate_df[result_bydate_df['returnPerc'] > 0]['returnPerc'].mean()
    avgretPcLoss = result_bydate_df[result_bydate_df['returnPerc'] < 0]['returnPerc'].mean()
    winByLossByretPc = abs(round(avgretPcWin / avgretPcLoss, 2))
    '''drawdowns'''
    maxddDuration = result_bydate_df['DDw_streak'].max()
    maxDurwithOutDD = result_bydate_df['DDl_streak'].max()
    maxddDate = result_bydate_df.loc[result_bydate_df['DDw_streak'] == maxddDuration, 'date'].values[0]
    maxDDPerc = result_bydate_df['drawDowns'].max()
    """collecting aggregated results for all days """
    avgRetPc = result_bydate_df['returnPerc'].mean()
    stdByRetPc = result_bydate_df['returnPerc'].std()
    avgRetDlrs = result_bydate_df['dayPL'].mean()
    stdPLDlrs = result_bydate_df['dayPL'].std()
    grossPL = result_bydate_df['dayPL'].sum()
    totRet, cagr = calcCAGR(result_bydate_df)
    MAR = round(cagr / maxDDPerc, 2)
    Sharpe = (avgRetPc / stdByRetPc) * math.sqrt(252)
    avgCapPerDay = result_bydate_df['startingCapital'].mean()
    capPerDayStd = result_bydate_df['startingCapital'].std()
    totDayswSignals = result_bydate_df.shape[0]
    totSymbols = result_bydate_df['symbolsCount'].sum()
    capImbAvg = result_bydate_df['spyHedgeCapital'].abs().mean()
    corrWdailySPYCO = result_bydate_df['dayPL'].corr(result_bydate_dfSPYCO['dayPL'])
    corrWdailySPYCORetPc = result_bydate_df['returnPerc'].corr(result_bydate_dfSPYCO['returnPerc'])
    symbolsPerSignalDay = round(totSymbols / totDayswSignals)
    sharesPerSignalDay = round(result_bydate_df['totalShares'].sum() / totDayswSignals)
    avgHedgeCapPerc = result_bydate_df['hedgeCapPerc'].mean()
    # totScreenedSymbols = result_bydate_df['totScreenedSymbols'].max()
    # totOrdersSubmitted = result_bydate_df['totOrdersSubmitted'].max()
    # totalFilledOrders = result_bydate_df['totalFilledOrders'].max()

    winDaysPc = ((result_bydate_df[result_bydate_df['dayPL'] > 0].shape[0]) / totDayswSignals) * 100
    results_df = pd.DataFrame(
        [{'avgRetPc': avgRetPc, 'stdByRetPc': stdByRetPc, 'avgRetDlrs': avgRetDlrs, 'stdPLDlrs': stdPLDlrs,
          'totSymbols': totSymbols,
          'grossPL': grossPL, 'maxDDPerc': maxDDPerc, 'maxddDuration': maxddDuration, 'CAGR': cagr, 'MAR': MAR,
          'Sharpe': Sharpe, 'avgCapPerDay': avgCapPerDay, 'capPerDayStd':capPerDayStd,'totDayswSignals': totDayswSignals,
          'winByLossPL': winByLossPL,
          'winDaysPc': winDaysPc, 'totRet': totRet,
          'maxLoss': maxLoss, 'maxProfit': maxProfit, 'maxWStreak': maxWStreak, 'maxLStreak': maxLStreak,
          'maxLStreakDate': maxLStreakDate, 'avgPLWindays': avgPLWindays, 'avgPLLossdays': avgPLLossdays,
          'avgretPcWin': avgretPcWin, 'avgretPcLoss': avgretPcLoss,
          'winByLossByretPc': winByLossByretPc,
          'maxDurwithOutDD': maxDurwithOutDD, 'maxddDate': maxddDate, 'corrWdailySPYCO': corrWdailySPYCO,
          'corrWdailySPYCORetPc': corrWdailySPYCORetPc, 'capImbAvg': capImbAvg,
          'symbolsPerSignalDay': symbolsPerSignalDay, 'sharesPerSignalDay': sharesPerSignalDay, 'avgHedgeCapPerc':avgHedgeCapPerc,
          # ,'totScreenedSymbols': totScreenedSymbols, 'totOrdersSubmitted': totOrdersSubmitted,
          # 'totalFilledOrders': totalFilledOrders
          }], index=None)
    return results_df.round(decimals=2)

def get_shares_perPosition(maxCapPerPos, maxRiskPerPos, tradeSide, df):

    dflowprice = df[(df['close'] >= 5) & (df['close'] <= 12)].copy(deep=True)
    dfmed1price = df[(df['close'] > 12) & (df['close'] <= 25)].copy(deep=True)
    dfmed2price = df[(df['close'] > 25) & (df['close'] <= 40)].copy(deep=True)
    dfhighprice = df[df['close'] > 40].copy(deep=True)

    maxCapPerPos = 5000
    dfhighprice = calc_shares(dfhighprice, maxCapPerPos, maxRiskPerPos, tradeSide)

    maxCapPerPos = 4500

    dfmed2price = calc_shares(dfmed2price, maxCapPerPos, maxRiskPerPos, tradeSide)

    maxCapPerPos = 3000

    dfmed1price = calc_shares(dfmed1price, maxCapPerPos, maxRiskPerPos, tradeSide)

    maxCapPerPos = 2000

    dflowprice = calc_shares(dflowprice, maxCapPerPos, maxRiskPerPos, tradeSide)

    df = dfhighprice.append([dflowprice, dfmed1price, dfmed2price], sort=False)
    df = df[df['shares'] >= 100]

    return df

def calc_shares(df, maxCapPerPos, maxRiskPerPos, tradeSide):
    if tradeSide == 'Long':
        df['shares'] = (maxCapPerPos / df['longsOrderPrice']).apply(np.floor)

    elif tradeSide == 'Short':
        df['shares'] = (maxCapPerPos / df['shortsOrderPrice']).apply(np.floor)
    if maxRiskPerPos > 0:
        df['shares'] = np.minimum((maxRiskPerPos / df['atr5']).apply(np.floor),
                                           df['shares'])
    return df

def get_NextTradesideCOPerc(df, trade):

    df['tradeSideNextSPYCOPerc'] = df['SPYNext_COPerc']
    if trade == 'Long':
        df['tradeSideNextCOPerc'] = df['next_COPerc']
        df['posSize'] = df['shares'] * df['longsOrderPrice']
    elif trade == 'Short':
        df['tradeSideNextCOPerc'] = df['next_COPerc'] * -1
        df['posSize'] = df['shares'] * df['shortsOrderPrice']
    return df

def get_positions_bydate(capital, df, tradeSide):
    df = get_NextTradesideCOPerc(df, tradeSide)

    df['cumPosSize'] = df.groupby('date')['posSize'].cumsum()
    df['screenedCountOnDt'] = df.groupby('date').transform('count')['posSize']

    df['totScreenedSymbols'] = df.shape[0]
    # df['totScrndSymsbyDate'] = df.groupby('date').size()
    df = df[df['cumPosSize'] <= capital]
    df['totOrdersSubmitted'] = df.shape[0]
    df['ordersCountOnDt'] = df.groupby('date').transform('count')['posSize']
    if tradeSide == 'Long':
        df = df[df['nextOpen'] <= df['longsOrderPrice']]
    elif tradeSide == 'Short':
        df = df[df['nextOpen'] >= df['shortsOrderPrice']]
    df['totalFilledOrders'] = df.shape[0]
    df['filledCountOnDt'] = df.groupby('date').transform('count')['posSize']
    # df['totFilledbyDate'] = df.groupby('date').count()
    df['dayPL'] = (df['tradeSideNextCOPerc'] * df['posSize']) / 100
    return df

def build_groupbyDate(df):
    longsbyDate_df = df[df['strategy'].str.contains("L")].copy(deep=True)
    shortsbyDate_df = df[df['strategy'].str.contains("S")].copy(deep=True)
    longsbyDate_df = longsbyDate_df.groupby('date', as_index=False)['posSize'].agg('sum').fillna(0)
    shortsbyDate_df = shortsbyDate_df.groupby('date', as_index=False)['posSize'].agg('sum').fillna(0)
    longsbyDate_df.columns = ['date', 'longsCapital']
    shortsbyDate_df.columns = ['date', 'shortsCapital']

    posistions_bydate_df = df.groupby('date', as_index=False).agg(
        {'ticker': 'count', 'dayPL': 'sum', 'SPYClose': 'mean', 'tradeSideNextSPYCOPerc': 'mean',
         'SPYNext_CCPerc': 'mean'
            , 'posSize': 'sum', 'shares': 'sum','totScreenedSymbols': 'max','totOrdersSubmitted': 'max','totalFilledOrders': 'max'
          ,'screenedCountOnDt': 'max','ordersCountOnDt': 'max','filledCountOnDt': 'max'}).round(decimals=2)

    posistions_bydate_df.columns = ['date', 'symbolsCount', 'dayPL', 'SPYClose', 'tradeSideNextSPYCOPerc',
                                    'SPYNext_CCPerc',
                                    'startingCapital', 'totalShares','totScreenedSymbols', 'totOrdersSubmitted', 'totalFilledOrders'
                                     ,'screenedCountOnDt', 'ordersCountOnDt', 'filledCountOnDt']
    posistions_bydate_df['daySpyCOPL'] = (
                posistions_bydate_df['tradeSideNextSPYCOPerc'] * posistions_bydate_df['startingCapital'] / 100)
    posistions_bydate_df = posistions_bydate_df.merge(longsbyDate_df, how='left')
    posistions_bydate_df = posistions_bydate_df.merge(shortsbyDate_df, how='left')
    posistions_bydate_df['longsCapital'] = posistions_bydate_df['longsCapital'].fillna(0)
    posistions_bydate_df['shortsCapital'] = posistions_bydate_df['shortsCapital'].fillna(0)

    return posistions_bydate_df

def main_method(hist_st_date, hist_end_date, source, hedge):
    print('Start Time:', dt.datetime.now(), hist_st_date, hist_end_date)
    sqlColumnsList = get_sqlcolumns_list()

    df, spy_df = get_stockhistory(sqlColumnsList, hist_st_date, hist_end_date, source)

    uniquedates_df = df.drop_duplicates(subset=['date'], keep='first').copy(deep=True)
    uniquedates_df.sort_values(['date'], inplace=True, ascending=True)

    uniquedates_df['SPY_CC_Perc'] = uniquedates_df['SPYNext_CCPerc'].shift(1)
    uniquedates_df['SPY_CO_Perc'] = uniquedates_df['SPYNext_COPerc'].shift(1)
    # uniquedates_df['SPY_OPC_Perc'] = uniquedates_df['SPYNext_OPCPerc'].shift(1)
    df['SPY_CC_Perc'] = df['date'].map(uniquedates_df.set_index('date')['SPY_CC_Perc'])
    df['SPY_CO_Perc'] = df['date'].map(uniquedates_df.set_index('date')['SPY_CO_Perc'])
    # df['SPY_OPC_Perc'] = df['date'].map(uniquedates_df.set_index('date')['SPY_OPC_Perc'])

    fvStats_df = df.copy(deep=True)
    df = df[df['next_COPerc']<100]
    df = df.dropna(subset=['sma100'])
    print('count after deleting anomolies', df.shape[0])
    priceConditions = (df['close'] >= 5) & (df['close'] <= 100)
    df = df[priceConditions]
    df = df[df['natr5']>1]
    print('count after filter', df.shape[0])
    capital = 50000
    maxCapPerPos = 4000
    maxRiskPerPos = 200

    df['shortsOrderPrice'] = df['FVNextOpen']
    df['longsOrderPrice'] = df['FVNextOpen']

    # df['shortsOrderPrice'] = df[['FVNextOpen', 'close']].apply(max, axis=1)
    # df['longsOrderPrice'] = df[['FVNextOpen', 'close']].apply(min, axis=1)

    '''Strategy S1 MR Short RSI Thrust'''
    setupCondition = ( (df['rsi3'] > 90) & (df['w_streak'] >= 4) )
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Short', setups_df)
    setups_df['strategy'] = 'S1'
    setups_df['rank'] = setups_df.groupby('date')['adx7'].rank('dense', ascending=False)
    setups_df['shortsOrderPrice'] = setups_df[['FVNextOpen', 'close']].apply(max, axis=1)
    diagnosis_df = setups_df
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfS1 = get_positions_bydate(capital, setups_df, 'Short')

    '''Strategy L1 OverSold Condition Reversal'''
    setupCondition = ( (df['rsi3'] < 30) & (df['close'] > df['ema100']) & (df['adx7'] > 45) )
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Long', setups_df)
    setups_df['strategy'] = 'L1'
    setups_df['rank'] = setups_df.groupby('date')['roc6'].rank('dense', ascending=True)
    diagnosis_df = diagnosis_df.append([setups_df], sort=False)
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfL1 = get_positions_bydate(capital, setups_df, 'Long')

    '''Strategy S2 MR Contrarian '''
    setupCondition =  ((df['l_streak'] >= 1)&(df['roc6'] > 10)&((df['SPY_CC_Perc'] >-.2)|(df['SPY_CO_Perc'] >-.2))&(df['close']>10))
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Short', setups_df)
    setups_df['strategy'] = 'S2'
    setups_df['rank'] = setups_df.groupby('date')['roc6'].rank('dense', ascending=False)
    setups_df['shortsOrderPrice'] = setups_df[['FVNextOpen', 'close']].apply(max, axis=1)
    diagnosis_df = diagnosis_df.append([setups_df], sort=False)
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfS2 = get_positions_bydate(capital, setups_df, 'Short')

    '''Strategy L2 Medium Trend Low Vol MR'''
    setupCondition = ( (df['natr50'] < 3)  & (df['close'] > df['sma100'])& (df['l_streak'] >=1 )& (df['adx7'] >65 ))
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Long', setups_df)
    setups_df['strategy'] = 'L2'
    setups_df['rank'] = setups_df.groupby('date')['rsi4'].rank('dense', ascending=True)
    setups_df['longsOrderPrice'] = setups_df[['FVNextOpen', 'close']].apply(min, axis=1)
    diagnosis_df = diagnosis_df.append([setups_df], sort=False)
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfL2 = get_positions_bydate(capital, setups_df, 'Long')

    '''Strategy S3 TR Sell Weakness'''
    setupCondition = ((df['close'] < df['sma5']) & (df['rsi4']<50) & (df['SPYClose']>df['SPYema100']))
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Short', setups_df)
    setups_df['strategy'] = 'S3'
    setups_df['rank'] = setups_df.groupby('date')['roc3'].rank('dense', ascending=True)
    setups_df['shortsOrderPrice'] = setups_df[['FVNextOpen', 'close']].apply(max, axis=1)
    diagnosis_df = diagnosis_df.append([setups_df], sort=False)
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfS3 = get_positions_bydate(capital, setups_df, 'Short')

    '''Strategy L3 Stay with the Trend'''
    setupCondition = ((df['rsi14'] > 70) & (df['rsi4'] > 80)&  (df['roc6'] > 1) )
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Long', setups_df)
    setups_df['strategy'] = 'L3'
    setups_df['rank'] = setups_df.groupby('date')['percBB5'].rank('dense', ascending=True)
    diagnosis_df = diagnosis_df.append([setups_df], sort=False)
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfL3 = get_positions_bydate(capital, setups_df, 'Long')

    '''Strategy S4 14 day trend continuation'''
    setupCondition = ( (df['adx14'] < 45)&(df['percBB14'] < .2)& (df['close'] < df['ema100'] )  )
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Short', setups_df)
    setups_df['strategy'] = 'S4'
    setups_df['rank'] = setups_df.groupby('date')['percBB14'].rank('dense', ascending=True)
    diagnosis_df = diagnosis_df.append([setups_df], sort=False)
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfS4 = get_positions_bydate(capital, setups_df, 'Short')

    '''Strategy L4 14 day trend continuation'''
    setupCondition = ((df['adx14'] < 45) & (df['percBB14'] > .2) & (df['close'] > df['ema100'])& (df['SPYClose']>df['SPYema100']))  # .15, 1.31
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Long', setups_df)
    setups_df['strategy'] = 'L4'
    setups_df['rank'] = setups_df.groupby('date')['percBB14'].rank('dense', ascending=False)
    diagnosis_df = diagnosis_df.append([setups_df], sort=False)
    setups_df['shortsOrderPrice'] = setups_df[['FVNextOpen', 'close']].apply(min, axis=1)
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfL4 = get_positions_bydate(capital, setups_df, 'Long')

    '''Strategy L5 short term trend continuation'''
    setupCondition = ( (df['natr50'] < 2)  & (df['rsi3'] > 65)& (df['w_streak'] >=1 )& (df['rsi14'] < df['rsi3'] ))
    setups_df = df[setupCondition].copy(deep=True)
    setups_df = get_shares_perPosition(maxCapPerPos, maxRiskPerPos, 'Long', setups_df)
    setups_df['strategy'] = 'L5'
    setups_df['rank'] = setups_df.groupby('date')['roc50'].rank('dense', ascending=True)
    diagnosis_df = diagnosis_df.append([setups_df], sort=False)
    setups_df = setups_df[setups_df['rank'] < 21]
    posistions_bydate_dfL5 = get_positions_bydate(capital, setups_df, 'Long')

    # diagnosis_df.to_csv('diagnosis.csv')
    # posistions_bydate_dfLongs = posistions_bydate_dfL5
    posistions_bydate_dfLongs = posistions_bydate_dfL1.append([posistions_bydate_dfL2, posistions_bydate_dfL3, posistions_bydate_dfL4, posistions_bydate_dfL5], sort = False)
    posistions_bydate_dfLongs = posistions_bydate_dfLongs.drop_duplicates(subset=['ticker', 'date'], keep='first')
    posistions_bydate_dfLongs.sort_values(['date', 'rank'], inplace=True, ascending=True)
    posistions_bydate_dfLongs = posistions_bydate_dfLongs[posistions_bydate_dfLongs['rank']<=20]
    posistions_bydate_dfLongs = get_positions_bydate(150000, posistions_bydate_dfLongs, 'Long')

    # posistions_bydate_dfShorts = posistions_bydate_dfS1.append([posistions_bydate_dfS2,posistions_bydate_dfS3 ], sort = False)
    posistions_bydate_dfShorts = posistions_bydate_dfS1.append([posistions_bydate_dfS2, posistions_bydate_dfS3,posistions_bydate_dfS4 ], sort = False)
    # posistions_bydate_dfShorts = posistions_bydate_dfS4
    posistions_bydate_dfShorts = posistions_bydate_dfShorts.drop_duplicates(subset=['ticker', 'date'], keep='first')
    posistions_bydate_dfShorts.sort_values(['date', 'rank'], inplace=True, ascending=True)
    posistions_bydate_dfShorts = posistions_bydate_dfShorts[posistions_bydate_dfShorts['rank']<=20]
    posistions_bydate_dfShorts = get_positions_bydate(150000, posistions_bydate_dfShorts, 'Short')

    posistions_bydate_df = posistions_bydate_dfLongs.append([posistions_bydate_dfShorts], sort = False)
    # posistions_bydate_df = posistions_bydate_dfShorts
    # posistions_bydate_df = posistions_bydate_dfLongs
    posistions_bydate_df = posistions_bydate_df.drop_duplicates(subset=['ticker', 'date'], keep=False)

    posistions_bydate_df = posistions_bydate_df.copy(deep=True)
    groupbyDate_df = build_groupbyDate(posistions_bydate_df)
    groupbyDate_dfSPYCO = groupbyDate_df.copy(deep=True)
    groupbyDate_dfSPYCO['dayPL'] = groupbyDate_dfSPYCO['daySpyCOPL']
    groupbyDate_dfSPYCO['shortsCapital'] = 0

    daily_metrics_df = build_daily_metrics(groupbyDate_df, hedge)
    daily_metrics_dfSPYCO = build_daily_metrics(groupbyDate_dfSPYCO)
    final_perf_metrics_df = get_final_perf(daily_metrics_df, daily_metrics_dfSPYCO)
 #   final_perf_metrics_dfSPYCO = get_final_perf(daily_metrics_dfSPYCO, daily_metrics_dfSPYCO)

    final_perf_metrics_df.to_csv('Performance_Results' + '_' + time.strftime("%Y%m%d-%H%M%S") + '.csv', index=False)
    print('Strategy Performance - HEDGE -', hedge,'\n', final_perf_metrics_df.to_string(index=False))
    # print('SPYCO Performance', '\n', final_perf_metrics_dfSPYCO.to_string(index=False))

    # get_SPYCC_BenchMark_Metrics(capital, daily_metrics_dfSPYCO, spy_df)

    # fvStats_df = get_FV_OpenStats(fvStats_df, hist_st_date, hist_end_date)
    # daily_metrics_df = daily_metrics_df.merge(fvStats_df, how='left')

    write_results_to_db(daily_metrics_df.round(decimals=2), spy_df.round(decimals=2),
                        posistions_bydate_df.round(decimals=2))
    # plot(daily_metrics_df)
    returns = daily_metrics_df[['date','dayPL']].copy()
    # plot(returns)
    by_month_pl = returns.set_index('date').resample('M').sum()
    by_month_pl.to_csv('by_month_returns2010.csv')
    #
    # print(by_month_pl)
    print('End Time:', dt.datetime.now(), hist_st_date, hist_end_date)

main_method('2018-01-01', '2020-05-02', 'SHARADAR', 'Y')
# main_method('1995-01-01', '2020-05-02', 'YAHOO', 'Y')
# main_method('2010-01-01', '2020-04-16', 'YAHOO', 'Y')

