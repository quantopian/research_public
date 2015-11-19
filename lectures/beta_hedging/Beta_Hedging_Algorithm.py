"""
Beta hedging

Author: David Edwards and James Christopher

This algorithm computes beta to the S&P 500 and attempts to maintain
a hedge for market neutrality. More information on beta hedging can
be found in the beta hedging lecture as part of the Quantopian
Lecture Series.

https://www.quantopian.com/lectures

   
This algorithm was developed as part of 
Quantopian's Lecture Series. Please direct any 
questions, feedback, or corrections to delaney@quantopian.com
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import math


def initialize(context):
    use_beta_hedging = True # Set to False to trade unhedged
    # Initialize a universe of liquid assets
    schedule_function(rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open())
    schedule_function(cancel_open_orders, date_rules.every_day(),
                      time_rules.market_close())
    
    if use_beta_hedging:
        # Call the beta hedging function one hour later to
        # make sure all of our orders have gone through.
        schedule_function(hedge_portfolio,
                          date_rule=date_rules.week_start(),
                          time_rule=time_rules.market_open(hours=1))
        
    # trading days used for beta calculation
    context.lookback = 150
    # Used to aviod purchasing any leveraged ETFs 
    context.dont_buys = security_lists.leveraged_etf_list
    # Current allocation per asset
    context.pct_per_asset = 0
    context.index = symbol('SPY')
    
    
def before_trading_start(context, data):
    # Number of stocks to find
    num_stocks = 100
    fundamental_df = get_fundamentals(
        query(
            # To add a metric. Start by typing "fundamentals."
            fundamentals.valuation_ratios.earning_yield,
            fundamentals.valuation.market_cap,
        )
        .filter(fundamentals.valuation.market_cap > 1e9)
        .order_by(fundamentals.valuation_ratios.earning_yield.desc())
        .limit(num_stocks)
    )
    context.eligible_assets = fundamental_df.columns
    
    update_universe(context.eligible_assets)
    
    
def rebalance(context, data):
    #  Sell assets no longer eligible.
    for asset in context.portfolio.positions:
        if asset in data and asset not in context.eligible_assets:
            if get_open_orders(asset):
                continue
            order_target_percent(asset, 0)
            
    #  Buy eligible assets.
    context.pct_per_asset = 1.0 / len(context.eligible_assets)
    for asset in context.eligible_assets:
        if asset in data:
            if get_open_orders(asset):
                continue
            order_target_percent(asset, context.pct_per_asset)
           
    
def hedge_portfolio(context, data):
    """
    This function places an order for "context.index" in the 
    amount required to neutralize the beta exposure of the portfolio.
    Note that additional leverage in the account is taken on, however,
    net market exposure is reduced.
    """
    factors = get_alphas_and_betas(context, data)
    beta_exposure = 0.0
    count = 0
    for asset in context.portfolio.positions:
        if asset in factors and asset != context.index:
            if not np.isnan(factors[asset].beta):
                beta_exposure += factors[asset].beta
                count += 1
    beta_hedge = -1.0 * beta_exposure / count
    dollar_amount = context.portfolio.portfolio_value * beta_hedge
    record(beta_hedge=beta_hedge)
    if not np.isnan(dollar_amount):
        order_target_value(context.index, dollar_amount)
    
    
def get_alphas_and_betas(context, data):
    """
    returns a dataframe of 'alpha' and 'beta' exposures 
    for each asset in the current universe.
    """
    prices = history(context.lookback, '1d', 'price', ffill=True)
    returns = prices.pct_change()[1:]
    index_returns = returns[context.index]
    factors = {}
    for asset in context.portfolio.positions:
        try:
            y = returns[asset]
            factors[asset] = linreg(index_returns, y)
        except:
            log.warn("[Failed Beta Calculation] asset = %s"%asset.symbol)
    return pd.DataFrame(factors, index=['alpha', 'beta'])
    
    
def linreg(x, y):
    # We add a constant so that we can also fit an intercept (alpha) to the model
    # This just adds a column of 1s to our data
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.params[0], model.params[1]


def handle_data(context, data):
    record(net_exposure=context.account.net_leverage,
           leverage=context.account.leverage,
           num_pos=len(context.portfolio.positions), 
           oo=len(get_open_orders()))
           

# Cancel all orders. This function is run at the end of everyday.
def cancel_open_orders(context, data):
    for security in get_open_orders():
        for order in get_open_orders(security):
            cancel_order(order)