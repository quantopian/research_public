"""This algorithm demonstrates the concept of long-short equity.
It uses two fundamental factors to rank equities in our universe.
It then longs the top of the ranking and shorts the bottom.
For information on long-short equity strategies, please see the corresponding
lecture on our lectures page:

https://www.quantopian.com/lectures

WARNING: These factors were selected because they worked in the past over the specific time period we choose.
We do not anticipate them working in the future. In practice finding your own factors is the hardest
part of developing any long-short equity strategy. This algorithm is meant to serve as a framework for testing your own ranking factors.

This algorithm was developed as part of
Quantopian's Lecture Series. Please direct any
questions, feedback, or corrections to max@quantopian.com
"""

import numpy as np
import pandas as pd
import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage 

from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline

from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.zacks import broker_ratings
from quantopian.pipeline.data.psychsignal import stocktwits

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
NUM_LONG_POSITIONS = 300
NUM_SHORT_POSITIONS = 300

# Here we define the maximum position size that can be held for any
# given stock. If you have a different idea of what these maximum 
# sizes should be, feel free to change them. Keep in mind that the
# optimizer needs some leeway in order to operate. Namely, if your
# maximum is too small, the optimizer may be overly-constrained.
MAX_SHORT_POSITION_SIZE = 2.0*1.0/(NUM_LONG_POSITIONS+NUM_SHORT_POSITIONS)
MAX_LONG_POSITION_SIZE = 2.0*1.0/(NUM_LONG_POSITIONS+NUM_SHORT_POSITIONS)

def make_pipeline():
    """
    Create and return our pipeline.

    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation.

    In particular, this function can be copy/pasted into research and run by itself.
    """
    
    # The factors we create here are based on broker recommendations data and a moving
    # average of sentiment data
    diff = (
        broker_ratings.rating_cnt_strong_buys.latest+broker_ratings.rating_cnt_mod_buys.latest -
        (broker_ratings.rating_cnt_strong_sells.latest+broker_ratings.rating_cnt_mod_sells.latest)
    )
    # Here we temper the diff between recommended buys and sells with a ratio of what
    # percentage of brokers actually rated a given security
    rat = broker_ratings.rating_cnt_with.latest/ \
        (broker_ratings.rating_cnt_with.latest+broker_ratings.rating_cnt_without.latest)
    alpha_signal = diff*rat
    
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
    )

    universe = QTradableStocksUS()

    # Construct a Factor representing the rank of each asset by our value
    # quality metrics. We aggregate them together here using simple addition
    # after zscore-ing them
    combined_factor = (
        alpha_signal.zscore() + sentiment_score.zscore()
    )

    # Build Filters representing the top and bottom NUM_POSITIONS stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    longs = combined_factor.top(NUM_LONG_POSITIONS, mask=universe)
    shorts = combined_factor.bottom(NUM_SHORT_POSITIONS, mask=universe)

    # The final output of our pipeline should only include
    # the top/bottom 300 stocks by our criteria
    long_short_screen = (longs | shorts)
    
    # Create pipeline
    pipe = Pipeline(
        columns = {
            'longs':longs,
            'shorts':shorts,
            'combined_factor':combined_factor
        },
        screen = long_short_screen
    )
    return pipe


def initialize(context):
    # Here we set our slippage and commisions. Set slippage 
    # and commission to zero to evaulate the signal-generating
    # ability of the algorithm independent of these additional
    # costs.
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=1, price_impact=0))
    context.spy = sid(8554)

    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')

    # attach the pipeline for the risk model factors that we 
    # want to neutralize in the optimization step
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(hours=0,minutes=30),
                      half_days=True)
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)


def before_trading_start(context, data):
    # Call algo.pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')

    # This dataframe will contain all of our risk loadings
    context.risk_loadings = algo.pipeline_output('risk_factors')

def recording_statements(context, data):
    # Plot the number of positions over time.
    record(num_positions=len(context.portfolio.positions))


# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    ### Optimize API
    pipeline_data = context.pipeline_data
    
    risk_loadings = context.risk_loadings

    ### Here we define our objective for the Optimize API. We have
    # selected MaximizeAlpha because we believe our combined factor
    # ranking to be proportional to expected returns. This routine
    # will optimize the expected return of our algorithm, going
    # long on the highest expected return and short on the lowest.
    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)
    
    ### Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())

    # Add the RiskModelExposure constraint to make use of the
    # default risk model constraints
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings
    )
    constraints.append(neutralize_risk_factors)
    
    # With this constraint we enforce that no position can make up
    # greater than MAX_SHORT_POSITION_SIZE on the short side and
    # no greater than MAX_LONG_POSITION_SIZE on the long side. This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    ### Put together all the pieces we defined above by passing
    # them into the algo.order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
