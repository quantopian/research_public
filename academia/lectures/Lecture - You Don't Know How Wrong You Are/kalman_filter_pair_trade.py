"""
Pairs Trading with Kalman Filters

Author: David Edwards 

This algorithm extends the Kalman Filtering pairs trading algorithm from a
previous lecture to support multiple pairs. In order to extend the idea,
the previous algorithm was factored into a class so several instances can be
created with different assets.


This algorithm was developed by David Edwards as part of 
Quantopian's 2015 summer lecture series. Please direct any 
questions, feedback, or corrections to dedwards@quantopian.com
"""


import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import statsmodels.api as sm


def initialize(context):
    # Quantopian backtester specific variables
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    set_symbol_lookup_date('2014-01-01')
    
    context.pairs = [
        KalmanPairTrade(symbol('STX'), symbol('WDC'),
                        initial_bars=300, freq='1m', delta=1e-3, maxlen=300),
        KalmanPairTrade(symbol('CBI'), symbol('JEC'),
                        initial_bars=300, freq='1m', delta=1e-3, maxlen=300),
        KalmanPairTrade(symbol('MAS'), symbol('VMC'),
                        initial_bars=300, freq='1m', delta=1e-3, maxlen=300),
        KalmanPairTrade(symbol('JPM'), symbol('C'),
                        initial_bars=300, freq='1m', delta=1e-3, maxlen=300),
        KalmanPairTrade(symbol('AON'), symbol('MMC'),
                        initial_bars=300, freq='1m', delta=1e-3, maxlen=300),
        KalmanPairTrade(symbol('COP'), symbol('CVX'),
                        initial_bars=300, freq='1m', delta=1e-3, maxlen=300),
       
    ]
    
    weight = 1.8 / len(context.pairs)
    for pair in context.pairs:
        pair.leverage = weight
        
    for minute in range(10, 390, 90):
        for pair in context.pairs:
            schedule_function(pair.trading_logic,
                              time_rule=time_rules.market_open(minutes=minute))
    

class KalmanPairTrade(object):

    def __init__(self, y, x, leverage=1.0, initial_bars=10, 
                 freq='1d', delta=1e-3, maxlen=3000):
        self._y = y
        self._x = x
        self.maxlen = maxlen
        self.initial_bars = initial_bars
        self.freq = freq
        self.delta = delta
        self.leverage = leverage
        self.Y = KalmanMovingAverage(self._y, maxlen=self.maxlen)
        self.X = KalmanMovingAverage(self._x, maxlen=self.maxlen)
        self.kf = None
        self.entry_dt = pd.Timestamp('1900-01-01', tz='utc')
        
    @property
    def name(self):
        return "{}~{}".format(self._y.symbol, self._x.symbol)

    def trading_logic(self, context, data):
        try:
            if self.kf is None:
                self.initialize_filters(context, data)
                return
            self.update()
            if get_open_orders(sid=self._x) or get_open_orders(sid=self._y):
                return
            spreads = self.mean_spread()

            zscore = (spreads[-1] - spreads.mean()) / spreads.std()

            reference_pos = context.portfolio.positions[self._y].amount

            now = get_datetime()
            if reference_pos:
                if (now - self.entry_dt).days > 20:
                    order_target(self._y, 0.0)
                    order_target(self._x, 0.0)
                    return
                # Do a PNL check to make sure a reversion at least covered trading costs
                # I do this because parameter drift often causes trades to be exited
                # before the original spread has become profitable.
                pnl = self.get_pnl(context, data)
                if zscore > -0.0 and reference_pos > 0 and pnl > 0:
                    order_target(self._y, 0.0)
                    order_target(self._x, 0.0)

                elif zscore < 0.0 and reference_pos < 0 and pnl > 0:
                    order_target(self._y, 0.0)
                    order_target(self._x, 0.0)

            else:
                if zscore > 1.5:
                    order_target_percent(self._y, -self.leverage / 2.)
                    order_target_percent(self._x, self.leverage / 2.)
                    self.entry_dt = now
                if zscore < -1.5:
                    order_target_percent(self._y, self.leverage / 2.)
                    order_target_percent(self._x, -self.leverage / 2.)
                    self.entry_dt = now
        except Exception as e:
            log.debug("[{}] {}".format(self.name, str(e)))

    def update(self):
        prices = np.log(history(bar_count=1, frequency='1m', field='price'))
        self.X.update(prices)
        self.Y.update(prices)
        self.kf.update(self.means_frame().iloc[-1])

    def mean_spread(self):
        means = self.means_frame()
        beta, alpha = self.kf.state_mean
        return means[self._y] - (beta * means[self._x] + alpha)


    def means_frame(self):
        mu_Y = self.Y.state_means
        mu_X = self.X.state_means
        return pd.DataFrame([mu_Y, mu_X]).T

            
    def initialize_filters(self, context, data):
        prices = np.log(history(self.initial_bars, self.freq, 'price'))
        self.X.update(prices)
        self.Y.update(prices)

        # Drops the initial 0 mean value from the kalman filter
        self.X.state_means = self.X.state_means.iloc[-self.initial_bars:]
        self.Y.state_means = self.Y.state_means.iloc[-self.initial_bars:]
        self.kf = KalmanRegression(self.Y.state_means, self.X.state_means,
                                   delta=self.delta, maxlen=self.maxlen)
    
    def get_pnl(self, context, data):
        x = self._x
        y = self._y
        prices = history(1, '1d', 'price').iloc[-1]
        positions = context.portfolio.positions
        dx = prices[x] - positions[x].cost_basis
        dy = prices[y] - positions[y].cost_basis
        return (positions[x].amount * dx +
                positions[y].amount * dy)
    
    
    
def handle_data(context, data):
    record(market_exposure=context.account.net_leverage,
           leverage=context.account.leverage)
    
    
class KalmanMovingAverage(object):
    """
    Estimates the moving average of a price process 
    via Kalman Filtering. 
    
    See http://pykalman.github.io/ for docs on the 
    filtering process. 
    """
    
    def __init__(self, asset, observation_covariance=1.0, initial_value=0,
                 initial_state_covariance=1.0, transition_covariance=0.05, 
                 initial_window=20, maxlen=3000, freq='1d'):
        
        self.asset = asset
        self.freq = freq
        self.initial_window = initial_window
        self.maxlen = maxlen
        self.kf = KalmanFilter(transition_matrices=[1],
                               observation_matrices=[1],
                               initial_state_mean=initial_value,
                               initial_state_covariance=initial_state_covariance,
                               observation_covariance=observation_covariance,
                               transition_covariance=transition_covariance)
        self.state_means = pd.Series([self.kf.initial_state_mean], name=self.asset)
        self.state_vars = pd.Series([self.kf.initial_state_covariance], name=self.asset)
        
        
    def update(self, observations):
        for dt, observation in observations[self.asset].iterkv():
            self._update(dt, observation)
        
    def _update(self, dt, observation):
        mu, cov = self.kf.filter_update(self.state_means.iloc[-1],
                                        self.state_vars.iloc[-1],
                                        observation)
        self.state_means[dt] = mu.flatten()[0]
        self.state_vars[dt] = cov.flatten()[0]
        if self.state_means.shape[0] > self.maxlen:
            self.state_means = self.state_means.iloc[-self.maxlen:]
        if self.state_vars.shape[0] > self.maxlen:
            self.state_vars = self.state_vars.iloc[-self.maxlen:]
        
        
class KalmanRegression(object):
    """
    Uses a Kalman Filter to estimate regression parameters 
    in an online fashion.
    
    Estimated model: y ~ beta * x + alpha
    """
    
    def __init__(self, initial_y, initial_x, delta=1e-5, maxlen=3000):
        self._x = initial_x.name
        self._y = initial_y.name
        self.maxlen = maxlen
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.expand_dims(
            np.vstack([[initial_x], [np.ones(initial_x.shape[0])]]).T, axis=1)
        
        self.kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                               initial_state_mean=np.zeros(2),
                               initial_state_covariance=np.ones((2, 2)),
                               transition_matrices=np.eye(2),
                               observation_matrices=obs_mat,
                               observation_covariance=1.0,
                               transition_covariance=trans_cov)
        state_means, state_covs = self.kf.filter(initial_y.values)
        self.means = pd.DataFrame(state_means, 
                                  index=initial_y.index, 
                                  columns=['beta', 'alpha'])
        self.state_cov = state_covs[-1]
        
    def update(self, observations):
        x = observations[self._x]
        y = observations[self._y]
        mu, self.state_cov = self.kf.filter_update(
            self.state_mean, self.state_cov, y, 
            observation_matrix=np.array([[x, 1.0]]))
        mu = pd.Series(mu, index=['beta', 'alpha'], 
                       name=observations.name)
        self.means = self.means.append(mu)
        if self.means.shape[0] > self.maxlen:
            self.means = self.means.iloc[-self.maxlen:]
        
    def get_spread(self, observations):
        x = observations[self._x]
        y = observations[self._y]
        return y - (self.means.beta[-1] * x + self.means.alpha[-1])
        
    @property
    def state_mean(self):
        return self.means.iloc[-1]
        