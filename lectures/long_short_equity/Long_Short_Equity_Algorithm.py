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
                          date_rule=date_rules.month_start(),
                          time_rule=time_rules.market_open(hours=1))
    # trading days used for beta calculation
    context.lookback = 250
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
    )
    
    fundamental_df
    yields = fundamental_df.ix['earning_yield']
    context.shorts = yields.tail(num_stocks/2)
    context.longs = yields.head(num_stocks/2)
    # fundamental_df = pd.concat([context.shorts, context.longs])
    all_assets = context.shorts.append(context.longs).index
    update_universe(all_assets)
    
def buy_assets(context, data):
    all_prices = history(1, '1d', 'price', ffill=True).dropna(axis=1)
    eligible_assets = [asset for asset in all_prices
                       if asset not in context.dont_buys 
                       and asset != context.index]
    pct_per_asset = 1.0 / max(len(eligible_assets), 1)
    context.pct_per_asset = pct_per_asset
    for asset in eligible_assets:
        # Some assets might cause a key error due to being delisted 
        # or some other corporate event so we use a try/except statement
        try:
            if get_open_orders(sid=asset):
                continue
            elif asset in context.shorts:
                order_target_percent(asset, -pct_per_asset)
            elif asset in context.longs:
                order_target_percent(asset, pct_per_asset)
        except:
            log.warn("[Failed Order] asset = %s"%asset.symbol)
    
def hedge_portfolio(context, data):
    """
    This function places an order for "context.index" in the 
    amount required to neutralize the beta exposure of the portfolio.
    """
    factors = get_alphas_and_betas(context, data)
    beta_exposure = 0.0
    count = 0
    for asset in context.portfolio.positions:
        if asset in factors and asset != context.index:
            position = context.portfolio.positions[asset].amount
            # Long the index if shorting the asset
            if not np.isnan(factors[asset].beta):
                beta_exposure += factors[asset].beta * np.copysign(1, position)
                count += 1
    beta_hedge = -1.0 * beta_exposure / max(count, 1)
    record(beta_hedge=beta_hedge)
    dollar_amount = context.portfolio.portfolio_value * beta_hedge
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
            X = returns[asset]
            factors[asset] = linreg(X, index_returns)
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
    
