import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, concatenate
from datetime import datetime

class KrakenApi:

    def __init__(self, trading_pair):
        self.trading_pair = trading_pair

    def get_all_trading_pairs(self):
        base_url = "https://api.kraken.com/0/public/"
        endpoint = "AssetPairs"

        response = requests.get(f"{base_url}{endpoint}")

        if response.status_code == 200:
            data = response.json()
            return data['result']
        else:
            print(f"Error: {response.status_code}")
            return None

    def get_order_book(self, count):
        base_url = "https://api.kraken.com/0/public/Depth"
        params = {
            "pair": self.trading_pair,
            "count": count
        }

        response = requests.get(base_url, params=params)
        data = response.json()
        print(data)
        if "result" in data and data["result"]:
            order_book = data["result"]
            return order_book
        else:
            print(f"Error: No data found for pair {self.trading_pair}")
            return None

    def get_last_trades(self, count, since=None):
        base_url = "https://api.kraken.com/0/public/"
        endpoint = "Trades"

        params = {
            'pair': self.trading_pair,
            'since': since,
            'count': count
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)

        if response.status_code == 200:
            data = response.json()
            return data['result'][self.trading_pair]
        else:
            print(f"Error: {response.status_code}")
            return None

    def get_latest_price(self):
        base_url = "https://api.kraken.com/0/public/"
        endpoint = "Ticker"

        params = {
            'pair': self.trading_pair
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)

        if response.status_code == 200:
            data = response.json()
            if 'result' in data and self.trading_pair in data['result']:
                # The price is typically available in the 'c' (close) field
                latest_price = data['result'][self.trading_pair]['c'][0]
                return float(latest_price)
            else:
                print("Invalid response format or symbol not found.")
        else:
            print(f"Error: {response.status_code}")

    def get_ticker(self):
        base_url = "https://api.kraken.com/0/public/"
        endpoint = "Ticker"

        params = {
            'pair': self.trading_pair
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)

        if response.status_code == 200:
            data = response.json()
            return data['result'][self.trading_pair]
        else:
            print(f"Error: {response.status_code}")
            return None

