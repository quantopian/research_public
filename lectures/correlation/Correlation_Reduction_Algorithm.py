"""
Long/Short Cross-Sectional Momentum

Authors: David Edwards, Gilbert Wassermann

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
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.factors import CustomFactor


def initialize(context):

    context.lookback = 300
    context.return_window = 60
    context.long_leverage = 0.5
    context.short_leverage = -0.5
    context.reduce_correlation = True

    # There's bad data for this security so I ignore it
    context.ignores = [sid(7143)]
    schedule_function(rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open(minutes=20))

    # Record tracking variables at the end of each day.
    schedule_function(record_vars,
                      date_rules.every_day(),
                      time_rules.market_close())

    # Create our dynamic stock selector.
    attach_pipeline(pipeline(context), 'pipeline')


class mkt_cap(CustomFactor):
    """
    CustomFactor which returns the market cap
    of equities in the stock universe
    """

    inputs = [morningstar.valuation.market_cap]
    window_length = 1

    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]


def pipeline(context):

    # create pipeline
    pipe = Pipeline()

    # Add market cap
    pipe.add(mkt_cap(), 'Market Cap')

    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('pipeline')
    context.output[context.output['Market Cap'] > 1e8]
    context.output.sort(['Market Cap'], ascending=False, inplace=True)
    context.security_list = context.output.head(500).index

    prices = np.log(data.history(context.security_list,
                    'price', context.lookback, '1d')).dropna(axis=1)
    R = (prices / prices.shift(context.return_window)).dropna()
    R = R[np.isfinite(R[R.columns])].fillna(0)

    # Subtract the cross-sectional average out of each data point on each day.
    ranks = (R.T - R.T.mean()).T.mean()
    # Take the top and botton percentiles for the long and short baskets
    lower, upper = ranks.quantile([.05, .95])
    shorts = ranks[ranks <= lower]
    longs = ranks[ranks >= upper]

    # Get weights that reduce the correlation within each basket
    if context.reduce_correlation:
        daily_R = prices.pct_change().dropna()
        context.longs = get_reduced_correlation_weights(daily_R[longs.index])
        context.shorts = get_reduced_correlation_weights(daily_R[shorts.index])

    else:
        # Use even weights
        longs = longs.abs()
        context.longs /= longs.sum()

        shorts = shorts.abs()
        context.shorts /= shorts.sum()


def rebalance(context, data):
    print(context.security_list)
    for stock in context.security_list:
        if stock in context.ignores:
            continue
        try:
            if stock in context.shorts.index and data.can_trade(stock):
                order_target_percent(stock,
                                     context.short_leverage *
                                     context.shorts[stock])
            elif stock in context.longs.index and data.can_trade(stock):
                order_target_percent(stock,
                                     context.long_leverage *
                                     context.longs[stock])
            elif data.can_trade(stock):
                order_target(stock, 0)
        except:
            log.warn("[Failed Order] stock = %s" % stock.symbol)


def record_vars(context, data):
    leverage = context.account.leverage
    exposure = context.account.net_leverage
    record(leverage=leverage, exposure=exposure)

"""
def handle_data(context, data):

    print "LONG LIST"
    log.info(context.longs)

    print "SHORT LIST"
    log.info(context.shorts)
"""

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
