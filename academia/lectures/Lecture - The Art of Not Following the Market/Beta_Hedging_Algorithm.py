import pandas as pd
import numpy as np
import statsmodels.api as sm
import math


def initialize(context):
    use_beta_hedging = True # Set to False to trade unhedged
    # Initialize a universe of liquid assets
    schedule_function(buy_assets,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open())
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
    
def before_trading_start(context):
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
    update_universe(fundamental_df)
    
def buy_assets(context, data):
    all_prices = history(1, '1d', 'price', ffill=True)
    eligible_assets = [asset for asset in all_prices
                       if asset not in context.dont_buys 
                       and asset != context.index]
    pct_per_asset = 1.0 / len(eligible_assets)
    context.pct_per_asset = pct_per_asset
    for asset in eligible_assets:
        # Some assets might cause a key error due to being delisted 
        # or some other corporate event so we use a try/except statement
        try:
            if get_open_orders(sid=asset):
                continue
            order_target_percent(asset, pct_per_asset)
        except:
            log.warn("[Failed Order] asset = %s"%asset.symbol)
           
    
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
           leverage=context.account.leverage)
    