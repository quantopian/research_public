"""
Long/Short Cross-Sectional Momentum

Author: David Edwards 

This algorithm implements a long/short strategy that looks at
an N day window of M day returns. It buys the assets that have 
had the most consistent returns relative to other assets and shorts
assets that have consistently underperformed.

Asset weights can be optionally scaled to reduce the correlations within
each basket. Even weights are used if the correlation reduction is not used.

reference for minimum correlation algorithm.
    http://cssanalytics.com/doc/MCA%20Paper.pdf
   
This algorithm was developed by David Edwards as part of 
Quantopian's 2015 summer lecture series. Please direct any 
questions, feedback, or corrections to dedwards@quantopian.com
"""

import numpy as np
import scipy
import pandas as pd


def initialize(context):
    
    context.lookback = 300
    context.return_window = 50
    context.longleverage = 0.5
    context.shortleverage = -0.5
    context.reduce_correlation = True
    
    # There's bad data for this security so I ignore it
    context.ignores = [sid(7143)]
    schedule_function(trade,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open(minutes=20))
    

    
def handle_data(context, data):
    leverage=context.account.leverage
    exposure=context.account.net_leverage
    record(leverage=leverage, exposure=exposure)
    
def trade(context, data):
    
    prices = np.log(history(context.lookback, '1d', 'price').dropna(axis=1))
    
    R = (prices / prices.shift(context.return_window)).dropna()
    
    # Subtract the cross-sectional average out of each data point on each day. 
    ranks = (R.T - R.T.mean()).T.mean()
    # Take the top and botton percentiles for the long and short baskets 
    lower, upper = ranks.quantile([.05, .95])
    shorts = ranks[ranks <= lower]
    longs = ranks[ranks >= upper]
    
    # Get weights that reduce the correlation within each basket
    if context.reduce_correlation:
        daily_R = prices.pct_change().dropna()
        longs = get_reduced_correlation_weights(daily_R[longs.index])
        shorts = get_reduced_correlation_weights(daily_R[shorts.index])
    else:
        # Use even weights
        longs = longs.abs()
        longs /= longs.sum()
        
        shorts = shorts.abs()
        shorts /= shorts.sum()
        
    for stock in data:
        if stock in context.ignores:
            continue
        try:
            if stock in shorts.index:
                order_target_percent(stock, 
                                     context.shortleverage * shorts[stock])
            elif stock in longs.index:
                order_target_percent(stock, 
                                     context.longleverage * longs[stock])
            else:
                order_target(stock, 0)
        except:
            log.warn("[Failed Order] stock = %s"%stock.symbol)
            


def get_reduced_correlation_weights(returns, risk_adjusted=True):
    """
    Implementation of minimum correlation algorithm.
    ref: http://cssanalytics.com/doc/MCA%20Paper.pdf
    
    :Params:
        :returns <Pandas DataFrame>:Timeseries of asset returns
        :risk_adjusted <boolean>: If True, asset weights are scaled
                                  by their standard deviations
    """
    correlations = returns.corr()
    adj_correlations = get_adjusted_cor_matrix(correlations)
    initial_weights = adj_correlations.T.mean()

    ranks = initial_weights.rank()
    ranks /= ranks.sum()

    weights = adj_correlations.dot(ranks)
    weights /= weights.sum()

    if risk_adjusted:
        weights = weights / returns.std()
        weights /= weights.sum()
    return weights

def get_adjusted_cor_matrix(cor):
    values = cor.values.flatten()
    mu = np.mean(values)
    sigma = np.std(values)
    distribution = scipy.stats.norm(mu, sigma)
    return 1 - cor.apply(lambda x: distribution.cdf(x))


def before_trading_start(context):
    num_stocks = 500
    fundamental_df = get_fundamentals(
        query(
            # To add a metric. Start by typing "fundamentals."
            fundamentals.valuation.market_cap,
        )
        .filter(fundamentals.valuation.market_cap > 1e8)
        .order_by(fundamentals.valuation.market_cap.desc())
        .limit(num_stocks)
    )
    update_universe(fundamental_df)
    
    