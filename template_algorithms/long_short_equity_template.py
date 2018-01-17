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

from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume, RollingLinearRegressionOfReturns
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.classifiers.fundamentals import Sector  

import numpy as np
import pandas as pd

import quantopian.optimize as opt
from quantopian.pipeline.experimental import QTradableStocksUS, risk_loading_pipeline

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
NUM_LONG_POSITIONS = 300
NUM_SHORT_POSITIONS = 300

# Here we define the maximum position size that can be held for any
# given stock. If you have a different idea of what these maximum 
# sizes should be, feel free to change them. Keep in mind that the
# optimizer needs some leeway in order to operate. Namely, if your
# maximum is too small, the optimizer may be overly-constrained.
MAX_SHORT_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
MAX_LONG_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

def make_pipeline():
    """
    Create and return our pipeline.

    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation.

    In particular, this function can be copy/pasted into research and run by itself.
    """
    
    # By appending .latest to the imported morningstar data, we get builtin Factors
    # so there's no need to define a CustomFactor
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    
    # Classify all securities by sector so that we can enforce sector neutrality later
    sector = Sector()
    
    # Screen out non-desirable securities by defining our universe. 
    # Removes ADRs, OTCs, non-primary shares, LP, etc.
    # Also sets a minimum $500MM market cap filter and $5 price filter
    mkt_cap_filter = Fundamentals.market_cap.latest >= 500000000    
    price_filter = USEquityPricing.close.latest >= 5
    universe = QTradableStocksUS() & price_filter & mkt_cap_filter

    # Construct a Factor representing the rank of each asset by our value
    # quality metrics. We aggregate them together here using simple addition
    # after zscore-ing them
    combined_rank = (
        value.zscore() +
        quality.zscore()
    )

    # Build Filters representing the top and bottom 150 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    longs = combined_rank.top(NUM_LONG_POSITIONS, mask=universe)
    shorts = combined_rank.bottom(NUM_SHORT_POSITIONS, mask=universe)

    # The final output of our pipeline should only include
    # the top/bottom 300 stocks by our criteria
    long_short_screen = (longs | shorts)
    
    # Create pipeline
    pipe = Pipeline(columns = {
        'longs':longs,
        'shorts':shorts,
        'combined_rank':combined_rank,
        'quality':quality,
        'value':value,
        'sector':sector
    },
    screen = long_short_screen)
    return pipe


def initialize(context):
    # Here we set our slippage and commisions. Set slippage 
    # and commission to zero to evaulate the signal-generating
    # ability of the algorithm independent of these additional
    # costs.
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=1, price_impact=0))
    context.spy = sid(8554)

    attach_pipeline(make_pipeline(), 'long_short_equity_template')

    # attach the pipeline for the risk model factors that we 
    # want to neutralize in the optimization step
    attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open(hours=0,minutes=30),
                      half_days=True)
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)


def before_trading_start(context, data):
    # Call pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = pipeline_output('long_short_equity_template')

    # This dataframe will contain all of our risk loadings
    context.risk_loadings = pipeline_output('risk_factors')

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
    objective = opt.MaximizeAlpha(pipeline_data.combined_rank)
    
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
    # them into the order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
    
