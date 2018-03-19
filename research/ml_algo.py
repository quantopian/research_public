# https://www.quantopian.com/posts/machine-learning-alpha-with-risk-constraints
# https://www.quantopian.com/posts/machine-learning-on-quantopian-part-3-building-an-algorithm?utm_campaign=machine-learning-on-quantopian-part-3-building-an-algorithm&utm_medium=email&utm_source=forums
from collections import OrderedDict
from time import time

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, metrics, linear_model

from quantopian.algorithm import (
    attach_pipeline,
    date_rules,
    order_optimal_portfolio,
    pipeline_output,
    record,
    schedule_function,
    set_commission,
    set_slippage,
    time_rules,
)
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.classifiers.fundamentals import Sector as _Sector
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import (
    CustomFactor,
    Returns,
    MACDSignal,
)
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from zipline.utils.numpy_utils import (
    repeat_first_axis,
    repeat_last_axis,
)

# If you have eventvestor, it's a good idea to screen out aquisition targets
# Comment out & ~IsAnnouncedAcqTarget() as well. You can also run this over
# the free period.
# from quantopian.pipeline.filters.eventvestor import IsAnnouncedAcqTarget

# Will be split 50% long and 50% short
N_STOCKS_TO_TRADE = 500

# Number of days to train the classifier on, easy to run out of memory here
ML_TRAINING_WINDOW = 252

# train on returns over N days into the future
PRED_N_FORWARD_DAYS = 5

# How often to trade, for daily, set to date_rules.every_day()
TRADE_FREQ = date_rules.week_start(days_offset=1) #date_rules.every_day()

class Sector(_Sector):
    window_safe = True


class MeanReversion1M(CustomFactor):
    inputs = (Returns(window_length=21),)
    window_length = 252

    def compute(self, today, assets, out, monthly_rets):
        np.divide(
            monthly_rets[-1] - np.nanmean(monthly_rets, axis=0),
            np.nanstd(monthly_rets, axis=0),
            out=out,
        )


class MoneyflowVolume5d(CustomFactor):
    inputs = (USEquityPricing.close, USEquityPricing.volume)

    # we need one more day to get the direction of the price on the first
    # day of our desired window of 5 days
    window_length = 6

    def compute(self, today, assets, out, close_extra, volume_extra):
        # slice off the extra row used to get the direction of the close
        # on the first day
        close = close_extra[1:]
        volume = volume_extra[1:]

        dollar_volume = close * volume
        denominator = dollar_volume.sum(axis=0)

        difference = np.diff(close_extra, axis=0)
        direction = np.where(difference > 0, 1, -1)
        numerator = (direction * dollar_volume).sum(axis=0)

        np.divide(numerator, denominator, out=out)


class PriceOscillator(CustomFactor):
    inputs = (USEquityPricing.close,)
    window_length = 252
    
    def compute(self, today, assets, out, close):
        four_week_period = close[-20:]
        np.divide(
            np.nanmean(four_week_period, axis=0),
            np.nanmean(close, axis=0),
            out=out,
        )
        out -= 1


class Trendline(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252

    _x = np.arange(window_length)
    _x_var = np.var(_x)

    def compute(self, today, assets, out, close):
        x_matrix = repeat_last_axis(
            (self.window_length - 1) / 2 - self._x,
            len(assets),
        )

        y_bar = np.nanmean(close, axis=0)
        y_bars = repeat_first_axis(y_bar, self.window_length)
        y_matrix = close - y_bars

        np.divide(
            (x_matrix * y_matrix).sum(axis=0) / self._x_var,
            self.window_length,
            out=out,
        )


class Volatility3M(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 63

    def compute(self, today, assets, out, rets):
        np.nanstd(rets, axis=0, out=out)


class AdvancedMomentum(CustomFactor):
    inputs = (USEquityPricing.close, Returns(window_length=126))
    window_length = 252

    def compute(self, today, assets, out, prices, returns):
        np.divide(
            (
                (prices[-21] - prices[-252]) / prices[-252] -
                prices[-1] - prices[-21]
            ) / prices[-21],
            np.nanstd(returns, axis=0),
            out=out,
        )


asset_growth_3m = Returns(
    inputs=[Fundamentals.total_assets],
    window_length=63,
)
asset_to_equity_ratio = (
    Fundamentals.total_assets.latest /
    Fundamentals.common_stock_equity.latest
)
capex_to_cashflows = (
    Fundamentals.capital_expenditure.latest /
    Fundamentals.free_cash_flow.latest
)

ebitda_yield = (
    (Fundamentals.ebitda.latest * 4) /
    USEquityPricing.close.latest
)
ebita_to_assets = (
    (Fundamentals.ebit.latest * 4) /
    Fundamentals.total_assets.latest
)
return_on_total_invest_capital = Fundamentals.roic.latest
mean_reversion_1m = MeanReversion1M()
macd_signal_10d = MACDSignal(
    fast_period=12,
    slow_period=26,
    signal_period=10,
)
moneyflow_volume_5d = MoneyflowVolume5d()
net_income_margin = Fundamentals.net_margin.latest
operating_cashflows_to_assets = (
    (Fundamentals.operating_cash_flow.latest * 4) /
    Fundamentals.total_assets.latest
)
price_momentum_3m = Returns(window_length=63)
price_oscillator = PriceOscillator()
trendline = Trendline()
returns_39w = Returns(window_length=215)
volatility_3m = Volatility3M()
advanced_momentum = AdvancedMomentum()


features = {
    'Asset Growth 3M': asset_growth_3m,
    'Asset to Equity Ratio': asset_to_equity_ratio,
    'Capex to Cashflows': capex_to_cashflows,
    'EBIT to Assets': ebita_to_assets,
    'EBITDA Yield': ebitda_yield,
    'MACD Signal Line': macd_signal_10d,
    'Mean Reversion 1M': mean_reversion_1m,
    'Moneyflow Volume 5D': moneyflow_volume_5d,
    'Net Income Margin': net_income_margin,
    'Operating Cashflows to Assets': operating_cashflows_to_assets,
    'Price Momentum 3M': price_momentum_3m,
    'Price Oscillator': price_oscillator,
    'Return on Invest Capital': return_on_total_invest_capital,
    '39 Week Returns': returns_39w,
    'Trendline': trendline,
    'Volatility 3m': volatility_3m,
    'Advanced Momentum': advanced_momentum,
}


def shift_mask_data(features,
                    labels,
                    n_forward_days,
                    lower_percentile,
                    upper_percentile):
    """Align features to the labels ``n_forward_days`` into the future and
    return the discrete, flattened features and masked labels.

    Parameters
    ----------
    features : np.ndarray
        A 3d array of (days, assets, feature).
    labels : np.ndarray
        The labels to predict.
    n_forward_days : int
        How many days into the future are we predicting?
    lower_percentile : float
        The lower percentile in the range [0, 100].
    upper_percentile : float
        The upper percentile in the range [0, 100].

    Returns
    -------
    selected_features : np.ndarray
        The flattened features that are not masked out.
    selected_labels : np.ndarray
        The labels that are not masked out.
    """

    # Slice off rolled elements
    shift_by = n_forward_days + 1
    aligned_features = features[:-shift_by]
    aligned_labels = labels[shift_by:]

    cutoffs = np.nanpercentile(
        aligned_labels,
        [lower_percentile, upper_percentile],
        axis=1,
    )
    discrete_labels = np.select(
        [
            aligned_labels <= cutoffs[0, :, np.newaxis],
            aligned_labels >= cutoffs[1, :, np.newaxis],
        ],
        [-1, 1],
    )

    # flatten the features per day
    flattened_features = aligned_features.reshape(
        -1,
        aligned_features.shape[-1],
    )

    # Drop stocks that did not move much, meaning they are in between
    # ``lower_percentile`` and ``upper_percentile``.
    mask = discrete_labels != 0

    selected_features = flattened_features[mask.ravel()]
    selected_labels = discrete_labels[mask]

    return selected_features, selected_labels


class ML(CustomFactor):
    """
    """
    train_on_weekday = 1

    def __init__(self, *args, **kwargs):
        CustomFactor.__init__(self, *args, **kwargs)

        self._imputer = preprocessing.Imputer()
        self._scaler = preprocessing.MinMaxScaler()
        self._classifier = linear_model.SGDClassifier(penalty='elasticnet')
        self.trained = False
        #ensemble.AdaBoostClassifier(
        #    random_state=1337,
        #    n_estimators=50,
        #)

    def _compute(self, *args, **kwargs):
        ret = CustomFactor._compute(self, *args, **kwargs)

        # reset the day counter so that we will begin training at the start of
        # the next _compute call
        self._day_counter = -1

        return ret

    def _train_model(self, today, returns, inputs):
        log.info('training model for window starting on: {}', today)

        imputer = self._imputer
        scaler = self._scaler
        classifier = self._classifier

        features, labels = shift_mask_data(
            np.dstack(inputs),
            returns,
            n_forward_days=PRED_N_FORWARD_DAYS,
            lower_percentile=30,
            upper_percentile=70,
        )
        features = scaler.fit_transform(imputer.fit_transform(features))

        start = time()
        classifier.fit(features, labels)
        log.info('training took {} secs', time() - start)
        self.trained = True

    def _maybe_train_model(self, today, returns, inputs):
        if (today.weekday() == self.train_on_weekday) or not self.trained:
            self._train_model(today, returns, inputs)

    def compute(self, today, assets, out, returns, *inputs):
        # inputs is a list of factors, for example, assume we have 2 alpha
        # signals, 3 stocks, and a lookback of 2 days. Each element in the
        # inputs list will be data of one signal, so len(inputs) == 2. Then
        # each element will contain a 2-D array of shape [time x stocks]. For
        # example:
        # inputs[0]:
        # [[1, 3, 2], # factor 1 rankings of day t-1 for 3 stocks
        #  [3, 2, 1]] # factor 1 rankings of day t for 3 stocks
        # inputs[1]:
        # [[2, 3, 1], # factor 2 rankings of day t-1 for 3 stocks
        #  [1, 2, 3]] # factor 2 rankings of day t for 3 stocks
        self._maybe_train_model(today, returns, inputs)

        # Predict
        # Get most recent factor values (inputs always has the full history)
        last_factor_values = np.vstack([input_[-1] for input_ in inputs]).T
        last_factor_values = self._imputer.transform(last_factor_values)
        last_factor_values = self._scaler.transform(last_factor_values)

        # Predict the probability for each stock going up
        # (column 2 of the output of .predict_proba()) and
        # return it via assignment to out.
        #out[:] = self._classifier.predict_proba(last_factor_values)[:, 1]
        out[:] = self._classifier.predict(last_factor_values)


def make_ml_pipeline(universe, window_length=21, n_forward_days=5):
    pipeline_columns = OrderedDict()

    # ensure that returns is the first input
    pipeline_columns['Returns'] = Returns(
        inputs=(USEquityPricing.open,),
        mask=universe, window_length=n_forward_days + 1,
    )

    # rank all the factors and put them after returns
    pipeline_columns.update({
        k: v.rank(mask=universe) for k, v in features.items()
    })

    # Create our ML pipeline factor. The window_length will control how much
    # lookback the passed in data will have.
    pipeline_columns['ML'] = ML(
        inputs=pipeline_columns.values(),
        window_length=window_length + 1,
        mask=universe,
    )

    pipeline_columns['Sector'] = Sector()

    return Pipeline(screen=universe, columns=pipeline_columns)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    schedule_function(
        rebalance,
        TRADE_FREQ,
        time_rules.market_open(minutes=1),
    )

    # Record tracking variables at the end of each day.
    schedule_function(
        record_vars,
        date_rules.every_day(),
        time_rules.market_close(),
    )

    # Set up universe, alphas and ML pipline
    context.universe = QTradableStocksUS()
    # if you are using IsAnnouncedAcqTarget, uncomment the next line
    # context.universe &= IsAnnouncedAcqTarget()

    ml_pipeline = make_ml_pipeline(
        context.universe,
        n_forward_days=PRED_N_FORWARD_DAYS,
        window_length=ML_TRAINING_WINDOW,
    )
    # Create our dynamic stock selector.
    attach_pipeline(ml_pipeline, 'alpha_model')
    # Add the risk pipeline
    attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    context.past_predictions = {}
    context.hold_out_accuracy = 0
    context.hold_out_log_loss = 0
    context.hold_out_returns_spread_bps = 0


def evaluate_and_shift_hold_out(output, context):
    # Look at past predictions to evaluate classifier accuracy on hold-out data
    # A day has passed, shift days and drop old ones
    context.past_predictions = {
        k - 1: v
        for k, v in context.past_predictions.iteritems()
        if k > 0
    }

    if 0 in context.past_predictions:
        # Past predictions for the current day exist, so we can use todays'
        # n-back returns to evaluate them
        raw_returns = output['Returns']
        raw_predictions = context.past_predictions[0]

        # Join to match up equities
        returns, predictions = raw_returns.align(raw_predictions, join='inner')

        # Binarize returns
        returns_binary = returns > returns.median()
        predictions_binary = predictions > 0.5

        # Compute performance metrics
        context.hold_out_accuracy = metrics.accuracy_score(
            returns_binary.values,
            predictions_binary.values,
        )
        context.hold_out_log_loss = metrics.log_loss(
            returns_binary.values,
            predictions.values,
        )
        long_rets = returns[predictions_binary == 1].mean()
        short_rets = returns[predictions_binary == 0].mean()
        context.hold_out_returns_spread_bps = (long_rets - short_rets) * 10000

    # Store current predictions
    context.past_predictions[PRED_N_FORWARD_DAYS] = context.predicted_probs


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = pipeline_output('alpha_model')
    context.predicted_probs = output['ML']
    context.predicted_probs.index.rename(['date', 'equity'], inplace=True)
    
    context.risk_loadings = pipeline_output('risk_factors')

    evaluate_and_shift_hold_out(output, context)

    # These are the securities that we are interested in trading each day.
    context.security_list = context.predicted_probs.index


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """ 
        
    predictions = context.predicted_probs

    # Filter out stocks that can not be traded
    predictions = predictions.loc[data.can_trade(predictions.index)]
    # Select top and bottom N stocks
    n_long_short = min(N_STOCKS_TO_TRADE // 2, len(predictions) // 2)
    predictions_top_bottom = pd.concat([
        predictions.nlargest(n_long_short),
        predictions.nsmallest(n_long_short),
    ])

    # If classifier predicts many identical values, the top might contain
    # duplicate stocks
    predictions_top_bottom = predictions_top_bottom.iloc[
        ~predictions_top_bottom.index.duplicated()
    ]

    # predictions are probabilities ranging from 0 to 1
    predictions_top_bottom = (predictions_top_bottom - 0.5) * 2

    # pull in the risk factor loadings
    risk_loadings = context.risk_loadings
    
    # Setup Optimization Objective
    # Factor-weighted portfolio
    objective = opt.TargetWeights(predictions_top_bottom)

    # Setup Optimization Constraints
    constrain_gross_leverage = opt.MaxGrossExposure(1.0)
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -0.02,
        +0.02,
    )
    market_neutral = opt.DollarNeutral()

    if predictions_top_bottom.index.duplicated().any():
        log.debug(predictions_top_bottom.head())
    
    risk_neutral = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings
    )

    # Run the optimization. This will calculate new portfolio weights and
    # manage moving our portfolio toward the target.
    order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            market_neutral,
            risk_neutral
        ],
    )


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(
        leverage=context.account.leverage,
        hold_out_accuracy=context.hold_out_accuracy,
        hold_out_log_loss=context.hold_out_log_loss,
        hold_out_returns_spread_bps=context.hold_out_returns_spread_bps,
    )


def handle_data(context, data):
    pass
