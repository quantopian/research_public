"""
This algorithm demonstrates the concept of long-short equity.
It uses two fundamental factors to rank all equities.
It then longs the top of the ranking and shorts the bottom.
For information on long-short equity strategies, please see the corresponding lecture on

https://www.quantopian.com/lectures

The filter_universe filter is in place in order to only trade useful securities.

WARNING: These factors were selected because they worked in the past over the specific time period we choose.
We do not anticipate them working in the future. In practice finding your own factors is the hardest
part of developing any long-short equity strategy. This algorithm is meant to serve as a framework for testing your own ranking factors.

This algorithm was developed as part of
Quantopian's Lecture Series. Please direct any
questions, feedback, or corrections to delaney@quantopian.com
"""

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.classifiers.morningstar import Sector

import numpy as np
import pandas as pd


class Value(CustomFactor):

    inputs = [morningstar.income_statement.ebit,
              morningstar.valuation.enterprise_value]
    window_length = 1

    def compute(self, today, assets, out, ebit, ev):
        out[:] = ebit[-1] / ev[-1]


class Quality(CustomFactor):

    # Pre-declare inputs and window_length
    inputs = [morningstar.operation_ratios.roe,]
    window_length = 1

    def compute(self, today, assets, out, roe):
        out[:] = roe[-1]


def make_pipeline():
    """
    Create and return our pipeline.

    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation.

    In particular, this function can be copy/pasted into research and run by itself.
    """


    # Basic value and quality metrics.
    value = Value()
    quality = Quality()

    # screen out bad securities
    initial_screen = filter_universe()

    # Construct a Factor representing the average rank of each asset by our
    # value and quality metrics.
    # By applying a mask to the rank computations, we remove any stocks that failed
    # to meet our initial criteria **before** computing ranks.  This means that the
    # stock with rank 10.0 is the 10th-lowest stock that passed `initial_screen`.
    combined_rank = (
        value.rank(mask=initial_screen) +
        quality.rank(mask=initial_screen)
    )

    # Build Filters representing the top and bottom 200 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    longs = combined_rank.top(200)
    shorts = combined_rank.bottom(200)

    # The final output of our pipeline should only include
    # the top/bottom 200 stocks by our criteria
    pipe_screen = (longs | shorts)

    pipe_columns = {
        'longs':longs,
        'shorts':shorts,
        'combined_rank':combined_rank,
        'quality':quality,
        'value':value
    }

    # Create pipeline
    pipe = Pipeline(columns = pipe_columns, screen = pipe_screen)
    return pipe


def initialize(context):

    # Set slippage and commission to zero to evaulate the signal generating
    # ability of the algorithm
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))

    context.long_leverage = 0.50
    context.short_leverage = -0.50
    context.spy = sid(8554)

    attach_pipeline(make_pipeline(), 'ranking_example')

    # Used to avoid purchasing any leveraged ETFs
    context.dont_buys = security_lists.leveraged_etf_list

    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.month_start(days_offset=0),
                      time_rule=time_rules.market_open(hours=0,minutes=30),
                      half_days=True)

    # clean untradeable securities daily
    schedule_function(daily_clean,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=30))


def before_trading_start(context, data):
    # Call pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors which
    output = pipeline_output('ranking_example')
    ranks = output['combined_rank']

    long_ranks = ranks[output['longs']]
    short_ranks = ranks[output['shorts']]

    context.long_weights = (long_ranks / long_ranks.sum())
    log.info("Long Weights:")
    log.info(context.long_weights)

    context.short_weights = (short_ranks / short_ranks.sum())
    log.info("Short Weights:")
    log.info(context.short_weights)


def handle_data(context, data):

    # Record and plot the leverage, number of positions,
    # and expsoure of our portfolio over time.
    record(num_positions=len(context.portfolio.positions),
           exposure=context.account.net_leverage,
           leverage=context.account.leverage)
    pass


# called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):

    """
    Allocate our long/short portfolio based on the weights supplied by
    context.long_weights and context.short_weights.
    """

    # calculate how much of each stock to buy or hold
    long_pct = context.long_leverage / len(context.long_weights)
    short_pct = context.short_leverage / len(context.short_weights)

    # universe now contains just longs and shorts
    context.security_set = set(context.long_weights.index.union(context.short_weights.index))

    for stock in context.security_set:
        if data.can_trade(stock):
            if stock in context.long_weights.index:
                order_target_percent(stock, long_pct)
            elif stock in context.short_weights.index:
                order_target_percent(stock, short_pct)

    # close out stale positions
    daily_clean(context, data)


# make sure all untradeable securities are sold off each day
def daily_clean(context, data):

    for stock in context.portfolio.positions:
        if stock not in context.security_set and data.can_trade(stock):
            order_target_percent(stock, 0)


# filter out bad securities
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
    common_stock = morningstar.share_class_reference.security_type.latest.eq('ST00000001')
    not_lp_name = ~morningstar.company_reference.standard_name.latest.matches('.* L[\\. ]?P\.?$')
    not_lp_balance_sheet = morningstar.balance_sheet.limited_partnership.latest.isnull()
    have_data = morningstar.valuation.market_cap.latest.notnull()
    not_otc = ~morningstar.share_class_reference.exchange_id.latest.startswith('OTC')
    not_wi = ~morningstar.share_class_reference.symbol.latest.endswith('.WI')
    not_depository = ~morningstar.share_class_reference.is_depositary_receipt.latest
    primary_share = IsPrimaryShare()

    # Combine the above filters.
    tradable_filter = (common_stock & not_lp_name & not_lp_balance_sheet &
                       have_data & not_otc & not_wi & not_depository & primary_share)

    high_volume_tradable = (AverageDollarVolume(window_length=21,
                                                mask=tradable_filter).percentile_between(70, 100))

    screen = high_volume_tradable

    return screen
