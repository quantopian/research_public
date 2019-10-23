"""
Long/Short Cross-Sectional Momentum

Author: Gilbert Wassermann

This algorithm creates traditional value factors and standardizes
them using a synthetic S&P500. It then uses a 130/30 strategy to trade.

    https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf
    
Please direct any questions, feedback, or corrections to help@quantopian.com

The material on this website is provided for informational purposes only
and does not constitute an offer to sell, a solicitation to buy, or a 
recommendation or endorsement for any security or strategy, 
nor does it constitute an offer to provide investment advisory or other services by Quantopian.

In addition, the content of the website neither constitutes investment advice 
nor offers any opinion with respect to the suitability of any security or any specific investment. 
Quantopian makes no guarantees as to accuracy or completeness of the 
views expressed in the website. The views are subject to change, 
and may have become unreliable for various reasons, 
including changes in market conditions or economic circumstances.
"""

import numpy as np
import pandas as pd
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.data import morningstar as mstar

# Custom Factor 1 : Dividend Yield
class Div_Yield(CustomFactor):

    inputs = [morningstar.valuation_ratios.dividend_yield]
    window_length = 1

    def compute(self, today, assets, out, d_y):
        out[:] = d_y[-1]

        
# Custom Factor 2 : P/B Ratio
class Price_to_Book(CustomFactor):

    inputs = [morningstar.valuation_ratios.pb_ratio]
    window_length = 1

    def compute(self, today, assets, out, p_b_r):
        out[:] = -p_b_r[-1]

        
# Custom Factor 3 : Price to Trailing 12 Month Sales       
class Price_to_TTM_Sales(CustomFactor):
    inputs = [morningstar.valuation_ratios.ps_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, ps):
        out[:] = -ps[-1]

        
# Custom Factor 4 : Price to Trailing 12 Month Cashflow
class Price_to_TTM_Cashflows(CustomFactor):
    inputs = [morningstar.valuation_ratios.pcf_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, pcf):
        out[:] = -pcf[-1] 
 

# This factor creates the synthetic S&P500
class SPY_proxy(CustomFactor):
    inputs = [morningstar.valuation.market_cap]
    window_length = 1
    
    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]
        
        
# This pulls all necessary data in one step
def Data_Pull():
    
    # create the pipeline for the data pull
    Data_Pipe = Pipeline()
    
    # create SPY proxy
    Data_Pipe.add(SPY_proxy(), 'SPY Proxy')

    # Div Yield
    Data_Pipe.add(Div_Yield(), 'Dividend Yield') 
    
    # Price to Book
    Data_Pipe.add(Price_to_Book(), 'Price to Book')
    
    # Price / TTM Sales
    Data_Pipe.add(Price_to_TTM_Sales(), 'Price / TTM Sales')
    
    # Price / TTM Cashflows
    Data_Pipe.add(Price_to_TTM_Cashflows(), 'Price / TTM Cashflow')
        
    return Data_Pipe


# function to filter out unwanted values in the scores
def filter_fn(x):
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x   


def standard_frame_compute(df):
    """
    Standardizes the Pipeline API data pull
    using the S&P500's means and standard deviations for
    particular CustomFactors.

    parameters
    ----------
    df: numpy.array
        full result of Data_Pull

    returns
    -------
    numpy.array
        standardized Data_Pull results
        
    numpy.array
        index of equities
    """
    
    # basic clean of dataset to remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # need standardization params from synthetic S&P500
    df_SPY = df.sort(columns='SPY Proxy', ascending=False)

    # create separate dataframe for SPY
    # to store standardization values
    df_SPY = df_SPY.head(500)
    
    # get dataframes into numpy array
    df_SPY = df_SPY.as_matrix()
    
    # store index values
    index = df.index.values
    
    # turn iinto a numpy array for speed
    df = df.as_matrix()
    
    # create an empty vector on which to add standardized values
    df_standard = np.empty(df.shape[0])
    
    for col_SPY, col_full in zip(df_SPY.T, df.T):
        
        # summary stats for S&P500
        mu = np.mean(col_SPY)
        sigma = np.std(col_SPY)
        col_standard = np.array(((col_full - mu) / sigma)) 

        # create vectorized function (lambda equivalent)
        fltr = np.vectorize(filter_fn)
        col_standard = (fltr(col_standard))
        
        # make range between -10 and 10
        col_standard = (col_standard / df.shape[1])
        
        # attach calculated values as new row in df_standard
        df_standard = np.vstack((df_standard, col_standard))
     
    # get rid of first entry (empty scores)
    df_standard = np.delete(df_standard,0,0)
    
    return (df_standard, index)


def composite_score(df, index):
    """
    Summarize standardized data in a single number.

    parameters
    ----------
    df: numpy.array
        standardized results
        
    index: numpy.array
        index of equities
        
    returns
    -------
    pandas.Series
        series of summarized, ranked results

    """

    # sum up transformed data
    df_composite = df.sum(axis=0)
    
    # put into a pandas dataframe and connect numbers
    # to equities via reindexing
    df_composite = pd.Series(data=df_composite,index=index)
    
    # sort descending
    df_composite.sort(ascending=False)

    return df_composite


def initialize(context):   
    
    # get data from pipeline
    data_pull = Data_Pull()
    attach_pipeline(data_pull,'Data')
    
    # filter out bad stocks for universe
    mask = filter_universe()
    data_pull.set_screen(mask)
    
    # set leverage ratios for longs and shorts
    context.long_leverage = 1.3
    context.short_leverage = -0.3
    
    # at the start of each moth, run the rebalancing function
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open(minutes=30))
    
    # clean untradeable securities daily
    schedule_function(daily_clean,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=30))    
    
    # record variables
    schedule_function(record_vars,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close())
    pass


# called before every day of trading
def before_trading_start(context, data):
    
    # apply the logic to the data pull in order to get a ranked list of equities
    context.output = pipeline_output('Data')
    context.output, index = standard_frame_compute(context.output)
    context.output = composite_score(context.output, index)
    
    # create lists of stocks on which to go long and short
    context.long_set = set(context.output.head(26).index)
    context.short_set =  set(context.output.tail(6).index)
    
# log long and short equities and their corresponding composite scores
def handle_data(context, data):
    """
    print "LONG LIST"
    log.info(context.long_set)  
    
    print "SHORT LIST"
    log.info(context.short_set)
    """
    pass


# called at the start of every month in order to rebalance the longs and shorts lists
def rebalance(context, data):
    
    # calculate how much of each stock to buy or hold
    long_pct = context.long_leverage / len(context.long_set)
    short_pct = context.short_leverage / len(context.short_set)
   
    # universe now contains just longs and shorts
    context.security_set = set(context.long_set.union(context.short_set))

    for stock in context.security_set:
        if data.can_trade(stock):
            if stock in context.long_set:
                order_target_percent(stock, long_pct)
            elif stock in context.short_set:
                order_target_percent(stock, short_pct)

    # close out stale positions    
    daily_clean(context, data)

# make sure all untradeable securities are sold off each day
def daily_clean(context, data):
    
    for stock in context.portfolio.positions:
        if stock not in context.security_set and data.can_trade(stock):
            order_target_percent(stock, 0)
    
def record_vars(context, data):

    # number of long and short positions. Even in minute mode, only the end-of-day
    # leverage is plotted.

    shorts = longs = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount < 0:
            shorts += 1
        elif position.amount > 0:
            longs += 1
    record(leverage=context.account.leverage, short_count=shorts, long_count=longs,
          exposure=context.account.net_leverage)
    
def filter_universe():  
    """
    9 filters:
        1. common stock
        2 & 3. not limited partnership - name and database check
        4. database has fundamental data
        5. not over the counter
        6. not when issued
        7. not depository receipts
        8. primary share
        9. high dollar volume
    Check Scott's notebook for more details.
    """
    common_stock = mstar.share_class_reference.security_type.latest.eq('ST00000001')
    not_lp_name = ~mstar.company_reference.standard_name.latest.matches('.* L[\\. ]?P\.?$')
    not_lp_balance_sheet = mstar.balance_sheet.limited_partnership.latest.isnull()
    have_data = mstar.valuation.market_cap.latest.notnull()
    not_otc = ~mstar.share_class_reference.exchange_id.latest.startswith('OTC')
    not_wi = ~mstar.share_class_reference.symbol.latest.endswith('.WI')
    not_depository = ~mstar.share_class_reference.is_depositary_receipt.latest
    primary_share = IsPrimaryShare()
    
    # Combine the above filters.
    tradable_filter = (common_stock & not_lp_name & not_lp_balance_sheet &
                       have_data & not_otc & not_wi & not_depository & primary_share)
    
    high_volume_tradable = (AverageDollarVolume(window_length=21,
                                                mask=tradable_filter).percentile_between(70, 100))

    screen = high_volume_tradable
    
    return screen
