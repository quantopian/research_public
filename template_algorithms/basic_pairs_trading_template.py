"""
This is a basic pairs trading algorithm for use at Quantopian Workshops.
WARNING: THIS IS A LEARNING EXAMPLE ONLY. DO NOT TRY TO TRADE SOMETHING THIS SIMPLE.
https://www.quantopian.com/workshops
https://www.quantopian.com/lectures
By Delaney Granizo-Mackenzie
"""
import numpy as np
 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    # Check status of the pair every day 2 minutes before we rebalance
    # The 2 minutes is just because we want to be safe, and 1 minutes
    # is cutting it close
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=60))
    
    context.stock1 = symbol('ABGB')
    context.stock2 = symbol('FSLR')
    
    # Our threshold for trading on the z-score
    context.entry_threshold = 0.2
    context.exit_threshold = 0.1
    
    # Moving average lengths
    context.long_ma_length = 30
    context.short_ma_length = 1
    
    # Flags to tell us if we're currently in a trade
    context.currently_long_the_spread = False
    context.currently_short_the_spread = False


def check_pair_status(context, data):
    
    # For notational convenience
    s1 = context.stock1
    s2 = context.stock2
    
    # Get pricing history
    prices = data.history([s1, s2], "price", context.long_ma_length, '1d')
    
    # Try debugging me here to see what the price
    # data structure looks like
    # To debug, click on the line number to the left of the
    # next command. Line numbers on blank lines or comments
    # won't work.
    short_prices = prices.iloc[-context.short_ma_length:]
    
    # Get the long mavg
    long_ma = np.mean(prices[s1] - prices[s2])
    # Get the std of the long window
    long_std = np.std(prices[s1] - prices[s2])
    
    
    # Get the short mavg
    short_ma = np.mean(short_prices[s1] - short_prices[s2])
    
    # Compute z-score
    if long_std > 0:
        zscore = (short_ma - long_ma)/long_std
    
        # Our two entry cases
        if zscore > context.entry_threshold and \
            not context.currently_short_the_spread:
            order_target_percent(s1, -0.5) # short top
            order_target_percent(s2, 0.5) # long bottom
            context.currently_short_the_spread = True
            context.currently_long_the_spread = False
            
        elif zscore < -context.entry_threshold and \
            not context.currently_long_the_spread:
            order_target_percent(s1, 0.5) # long top
            order_target_percent(s2, -0.5) # short bottom
            context.currently_short_the_spread = False
            context.currently_long_the_spread = True
            
        # Our exit case
        elif abs(zscore) < context.exit_threshold:
            order_target_percent(s1, 0)
            order_target_percent(s2, 0)
            context.currently_short_the_spread = False
            context.currently_long_the_spread = False
        
        record('zscore', zscore)