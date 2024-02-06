import json
import os

import joblib
import pandas as pd
from pandas import np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model

from crypto.kraken_api import KrakenApi
import time

class dataCreation:
    def __init__(self, world_instance, trading_pair):
        self.world_instance = world_instance
        self.trading_data = KrakenApi(trading_pair)
        self.result_dict = {}
        self.list_of_dicts = []
        self.models_path = r'C:\Users\Filip\PycharmProjects\tensorEnv\crypto\model'
        os.makedirs(self.models_path, exist_ok=True)
        self.models = {}


    def retrieve_data(self, item):
        # Retrieve data from the World instance
        data = self.world_instance.get_data(item)
        print(f"Item from World: {data}")
        return data

    def collect_process_data(self):
        # Convert dictionaries to DataFrames
        sell_data = self.retrieve_data('asks')
        final_sell_data = self.process_trade_data(sell_data, 'sell')
        buy_data = self.retrieve_data('bids')
        final_buy_data = self.process_trade_data(buy_data, 'buy')

        lowest_sell_data = self.retrieve_data('ticker')['a']
        final_lowest_sell_data = self.process_ticker_data(lowest_sell_data, 'sell')
        highest_buy_data = self.retrieve_data('ticker')['b']
        final_highest_buy_data = self.process_ticker_data(highest_buy_data, 'buy')

        current_price = self.retrieve_data('last price')

        prices = self.get_and_store_latest_price(1, 5)

        combined_data = {**final_sell_data, **final_buy_data, **final_lowest_sell_data, **final_highest_buy_data, **prices}

        print("Combined data", combined_data)
        return combined_data

    def collect_process_data_only_last_price(self):
        # Convert dictionaries to DataFrames
        sell_data = self.retrieve_data('asks')
        final_sell_data = self.process_trade_data(sell_data, 'sell')

        buy_data = self.retrieve_data('bids')
        final_buy_data = self.process_trade_data(buy_data, 'buy')

        current_price = self.retrieve_data('last price')
        current_price_dict = {'Current Price': current_price}

        lowest_sell_data = self.retrieve_data('ticker')['a']
        final_lowest_sell_data = self.process_ticker_data(lowest_sell_data, 'sell')
        highest_buy_data = self.retrieve_data('ticker')['b']
        final_highest_buy_data = self.process_ticker_data(highest_buy_data, 'buy')

        combined_data = {**final_sell_data, **final_buy_data, **final_lowest_sell_data, **final_highest_buy_data, **current_price_dict}

        print("Combined data", combined_data)
        return combined_data

    def process_trade_data(self, data, type):
        # Initialize variables to store overall average price and total volume
        total_average_price = 0
        total_total_volume = 0

        # Process the data and calculate overall average price and total volume
        for entry in data:
            price = float(entry[0])
            volume = float(entry[1])

            total_average_price += price
            total_total_volume += volume

        # Calculate the final overall average price
        overall_average_price = total_average_price / len(data)

        # Return the result as a dictionary containing average price and total volume
        result = {"Overall Average {} Price".format(type): overall_average_price, "Overall Total {} Volume".format(type): total_total_volume}
        print("result:", result)
        return result

    def process_ticker_data(self, data, type):

        if type == 'sell':
            # Return the result as a dictionary containing average price and total volume
            result = {"The lowest {} Price".format(type): float(data[0]),  "The volume at the lowest {} price".format(type):  float(data[2])}
            print("result:", result)
        else:
            result = {"The highest {} Price".format(type): float( data[0]),  "The volume at the highest {} price".format(type): float( data[2])}
            print("result:", result)
        return result

    def get_and_store_latest_price(self, sleep_time, latest_prices_amount):
        i = 0
        for _ in range(latest_prices_amount):
            current_price = self.trading_data.get_latest_price()
            key = f'current_price_{i}'
            self.result_dict[key] = current_price
            i = i + sleep_time
            time.sleep(sleep_time)

        return self.result_dict

    def save(self, data, trading_pair):

        # File path to store the data
        file_path = r'C:\Users\Filip\PycharmProjects\tensorEnv\crypto\data\combined_data_{}.json'.format(trading_pair)

        # Load existing data
        if not os.path.exists(file_path):
            self.list_of_dicts.append(data)
        else:
            with open(file_path, 'r') as file:
                self.list_of_dicts = json.load(file)
                self.list_of_dicts.append(data)
        # Save the combined data back to the file
        with open(file_path, 'w') as file:
            json.dump(self.list_of_dicts, file, indent=4)

        print("Data appended and saved to", file_path)

    def load_models(self):
        # self.models = {}
        targets = ["current_price_15", "current_price_30", "current_price_45", "current_price_60"]
        for target in targets:
            model_path = os.path.join(self.models_path, f'model_{target}.h5')
            scaler_path = os.path.join(self.models_path, f'scaler_{target}.joblib')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                self.models[target] = {"model": model, "scaler": scaler}
            else:
                print(f"Warning: Model or scaler not found for {target}")



