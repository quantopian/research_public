"""
Pairs Trading with Kalman Filters

Author: David Edwards 

This algorithm pair trades two solar companies. In order to demonstrate kalman filtering,
the price series are smoothed with a kalman filter and regression parameters are estimated
with another kalman filter.


This algorithm was developed by David Edwards as part of 
Quantopian's 2015 summer lecture series. Please direct any 
questions, feedback, or corrections to dedwards@quantopian.com
"""

import numpy as np
import pandas as pd
from pykalman import KalmanFilter


def initialize(context):
    # Quantopian backtester specific variables
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerTrade(cost=1))
    set_symbol_lookup_date('2014-01-01')
    context.X = KalmanMovingAverage(symbol('ABGB'))
    context.Y = KalmanMovingAverage(symbol('FSLR'))
    context.kf = None
    for minute in range(10, 390, 20):
        schedule_function(trade,
                          time_rule=time_rules.market_open(minutes=minute))


def trade(context, data):
    if context.kf is None:
        initialize_filters(context, data)
        return
    if get_open_orders():
        return
    prices = np.log(history(bar_count=1, frequency='1d', field='price'))
    context.X.update(prices)
    context.Y.update(prices)

    mu_Y = context.Y.state_means
    mu_X = context.X.state_means

    frame = pd.DataFrame([mu_Y, mu_X]).T

    context.kf.update(frame.iloc[-1])

    beta, alpha = context.kf.state_mean

    spreads = (mu_Y - (beta * mu_X + alpha)).tail(500)

    zscore = (spreads[-1] - spreads.mean()) / spreads.std()

    reference_pos = context.portfolio.positions[context.Y.asset].amount

    record(
        beta=beta,
        alpha=alpha,
        mean_spread=spreads[-1],
        zscore=zscore
    )

    if reference_pos:
        # Do a PNL check to make sure a reversion at least covered trading costs
        # I do this because parameter drift often causes trades to be exited 
        # before the original spread has become profitable. 
        pnl = get_pnl(context, data)
        if zscore > -0.75 and reference_pos > 0 and pnl > 10:
            order_target(context.Y.asset, 0.0)
            order_target(context.X.asset, 0.0)

        elif zscore < 0.75 and reference_pos < 0 and pnl > 10:
            order_target(context.Y.asset, 0.0)
            order_target(context.X.asset, 0.0)

    else:
        if zscore > 2.0:
            order_target_percent(context.Y.asset, -0.5)
            order_target_percent(context.X.asset, 0.5)
        if zscore < -2.0:
            order_target_percent(context.Y.asset, 0.5)
            order_target_percent(context.X.asset, -0.5)


def initialize_filters(context, data):
    initial_bars = 10
    prices = np.log(history(initial_bars, '1d', 'price'))
    context.X.update(prices)
    context.Y.update(prices)

    # Drops the initial 0 mean value from the kalman filter
    context.X.state_means = context.X.state_means.iloc[-initial_bars:]
    context.Y.state_means = context.Y.state_means.iloc[-initial_bars:]

    context.kf = KalmanRegression(context.Y.state_means, context.X.state_means)


def get_pnl(context, data):
    x = context.X.asset
    y = context.Y.asset
    positions = context.portfolio.positions
    dx = data[x].price - positions[x].cost_basis
    dy = data[y].price - positions[y].cost_basis
    return (positions[x].amount * dx +
            positions[y].amount * dy)


def handle_data(context, data):
    record(market_exposure=context.account.net_leverage)


class KalmanMovingAverage(object):
    """
    Estimates the moving average of a price process 
    via Kalman Filtering. 
    
    See http://pykalman.github.io/ for docs on the 
    filtering process. 
    """

    def __init__(self, asset, observation_covariance=1.0, initial_value=0,
                 initial_state_covariance=1.0, transition_covariance=0.05,
                 initial_window=20, maxlen=5000, freq='1d'):
        self.asset = asset
        self.freq = freq
        self.maxlen = maxlen
        self.initial_window = initial_window

        self.kf = KalmanFilter(transition_matrices=[1],
                               observation_matrices=[1],
                               initial_state_mean=initial_value,
                               initial_state_covariance=initial_state_covariance,
                               observation_covariance=observation_covariance,
                               transition_covariance=transition_covariance)
        self.state_means = pd.Series([self.kf.initial_state_mean], name=self.asset)
        self.state_covs = pd.Series([self.kf.initial_state_covariance], name=self.asset)

    def update(self, observations):
        for dt, observation in observations[self.asset].iterkv():
            self._update(dt, observation)

    def _update(self, dt, observation):
        mu, cov = self.kf.filter_update(self.state_means.iloc[-1],
                                        self.state_covs.iloc[-1],
                                        observation)
        self.state_means[dt] = mu.flatten()[0]
        self.state_covs[dt] = cov.flatten()[0]
        if self.state_means.shape[0] > self.maxlen:
            self.state_means = self.state_means[-self.maxlen:]
        if self.state_covs.shape[0] > self.maxlen:
            self.state_covs = self.state_covs[-self.maxlen:]


class KalmanRegression(object):
    """
    Uses a Kalman Filter to estimate regression parameters 
    in an online fashion.
    
    Estimated model: y ~ beta * x + alpha
    """

    def __init__(self, initial_y, initial_x, delta=1e-5, maxlen=5000):
        self._x = initial_x.name
        self._y = initial_y.name
        trans_cov = delta / (1 - delta) * np.eye(2)
        self.maxlen = maxlen
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
        mu, self.state_cov = self.kf.filter_update(self.state_mean, self.state_cov, y,
                                                   observation_matrix=np.array([[x, 1.0]]))
        mu = pd.Series(mu, index=['beta', 'alpha'],
                       name=observations.name)
        self.means = self.means.append(mu).tail(self.maxlen)


    def get_spread(self, observations):
        x = observations[self._x]
        y = observations[self._y]
        return y - (self.means.beta * x + self.means.alpha)

    @property
    def state_mean(self):
        return self.means.iloc[-1]
