import alpaca_trade_api as tradeapi
import yfinance as yf
import talib
import numpy as np
import os
import time
import pytz
from datetime import datetime

# This beta code has not been tested. 
# This code might work or it might not work. 

# Configure Alpaca API
API_KEY_ID = os.getenv('APCA_API_KEY_ID')
API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
API_BASE_URL = os.getenv('APCA_API_BASE_URL')

# Initialize Alpaca API
api = tradeapi.REST(API_KEY_ID, API_SECRET_KEY, API_BASE_URL)

# Define the trading symbol (e.g., SPY)
symbol = 'SPY'

# Define the time frame for market condition analysis (14 days)
time_frame = 14

# Define the threshold for bear/bull market determination (e.g., 5%)
threshold = 0.05

# Define the trading parameters
position_size = 1  # number of shares to trade
stop_loss = 0.02  # stop loss percentage
take_profit = 0.05  # take profit percentage

# Define the trading robot class
class AdaptiveTrader:
    def __init__(self, symbol, time_frame, threshold, position_size, stop_loss, take_profit):
        self.symbol = symbol
        self.time_frame = time_frame
        self.threshold = threshold
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.market_condition = None
        self.position = None

    def get_market_data(self):
        # Get historical price data for the symbol
        data = yf.download(self.symbol, period=str(self.time_frame) + 'd', interval='1d')
        closes = data['Close'].values

        # Calculate the price percentage change
        pct_change = np.diff(closes) / closes[:-1]

        # Determine the market condition (bear or bull)
        if np.mean(pct_change) < -self.threshold:
            self.market_condition = 'bear'
        elif np.mean(pct_change) > self.threshold:
            self.market_condition = 'bull'
        else:
            self.market_condition = 'neutral'

    def analyze_market(self):
        # Get the current market data
        self.get_market_data()

        # Analyze the market condition and adjust trading behavior
        if self.market_condition == 'bear':
            self.adjust_bear_market()
        elif self.market_condition == 'bull':
            self.adjust_bull_market()
        else:
            self.adjust_neutral_market()

    def adjust_bear_market(self):
        # Bear market behavior: reduce position size, tighten stop loss
        self.position_size *= 0.5
        self.stop_loss *= 0.5

    def adjust_bull_market(self):
        # Bull market behavior: increase position size, loosen stop loss
        self.position_size *= 1.5
        self.stop_loss *= 1.5

    def adjust_neutral_market(self):
        # Neutral market behavior: maintain position size and stop loss
        pass

    def execute_trade(self):
        # Get the current market price
        market_price = yf.download(self.symbol, period='1d', interval='1m')['Close'][-1]

        # Get account and position information
        account = api.get_account()
        position = api.get_position(self.symbol)
        day_trade_count = api.get_day_trade_count()

        # Check if we have a position
        if position is None:
            # Check if we have enough cash to buy
            if account.cash > market_price * self.position_size:
                api.submit_order(self.symbol, self.position_size, 'buy', 'market', 'day')
                print(f'Bought {self.symbol} at {market_price:.2f}')
        else:
            # Check if we need to close the position
            if position.side == 'long' and position.market_value > market_price and day_trade_count < 3 and market_price > position.avg_entry_price * 1.005:
                api.submit_order(self.symbol, self.position_size, 'sell', 'market', 'day')
                print(f'Sold {self.symbol} at {market_price:.2f}')

    def run(self):
        while True:
            try:
                # Print current date and time in Eastern Time, USA
                eastern_time = pytz.timezone('US/Eastern')
                current_time = datetime.now(eastern_time)
                print(f'Current Time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}')

                # Analyze market and execute trade
                self.analyze_market()
                self.execute_trade()

                # Print market condition and trading details
                print(f'Market Condition: {self.market_condition}')
                print(f'Position Size: {self.position_size}')
                print(f'Stop Loss: {self.stop_loss:.2f}%')
                print(f'Take Profit: {self.take_profit:.2f}%')

                # Get list of stocks to buy with their current prices
                stocks_to_buy = ['SPY', 'AAPL', 'GOOG']
                stock_prices = {}
                for stock in stocks_to_buy:
                    stock_prices[stock] = yf.download(stock, period='1d', interval='1m')['Close'][-1]
                print('Stocks to Buy:')
                for stock, price in stock_prices.items():
                    print(f'{stock}: {price:.2f}')

                # Get list of owned positions with their % change from purchase
                positions = api.list_positions()
                owned_positions = {}
                for position in positions:
                    owned_positions[position.symbol] = {'market_value': position.market_value, 'avg_entry_price': position.avg_entry_price}
                print('Owned Positions:')
                for symbol, position in owned_positions.items():
                    pct_change = (position['market_value'] - position['avg_entry_price']) / position['avg_entry_price'] * 100
                    print(f'{symbol}: {pct_change:.2f}%')

                # Print market condition details
                if self.market_condition == 'bear':
                    print('Market is bearish.')
                elif self.market_condition == 'bull':
                    print('Market is bullish.')
                else:
                    print('Market is neutral.')

            except Exception as e:
                print(f'Error: {e}')
                print('Restarting program in 5 seconds...')
                time.sleep(5)
                os.execl(sys.executable, sys.executable, *sys.argv)

# Create an instance of the trading robot
trader = AdaptiveTrader(symbol, time_frame, threshold, position_size, stop_loss, take_profit)

# Run the trading robot
trader.run()
