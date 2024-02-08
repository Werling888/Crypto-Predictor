from crypto.kraken_api import KrakenApi
from crypto.world_object import WorldObject
from crypto.LSTM_model import CryptoPricePredictor
from crypto.data_creation import dataCreation

class Main:

    def __init__(self):
        self.world_instance = WorldObject()
        self.trading_pair = "SOLEUR"
        self.trading_data = KrakenApi(self.trading_pair)

    def run_program(self):
        # Get all trading pairs
        all_trading_pairs = self.trading_data.get_all_trading_pairs()

        if all_trading_pairs:
            print("All available trading pairs:")
            for pair in all_trading_pairs:
                info = all_trading_pairs[pair]
                print(f"{pair}: {info['base']} to {info['quote']}")
        else:
            print("Failed to retrieve trading pairs.")

        # Get order book
        count = 1000
        order_book_data = self.trading_data.get_order_book(count)

        if order_book_data is not None:
            asks = order_book_data[self.trading_data.trading_pair]['asks']
            bids = order_book_data[self.trading_data.trading_pair]['bids']
            self.world_instance.set_data("asks", asks)
            self.world_instance.set_data("bids", bids)
            # sellers
            print(f"Top {count} bids: { self.world_instance.get_data('asks')}")
            # buyers
            print(f"Top {count} asks: {bids}")

            print('ask = sell')
            for ask in asks:
                print(f"Price: {ask[0]}, Volume: {ask[1]}, Timestamp: {ask[2]}")
            print('bid = buy')
            for bid in bids:
                print(f"Price: {bid[0]}, Volume: {bid[1]}, Timestamp: {bid[2]}")
        else:
            print("Error: Unable to retrieve order book data.")
            print("Error: Unable to retrieve order book data.")

        # Get last trades
        last_trades = self.trading_data.get_last_trades(count)

        if last_trades:
            self.world_instance.set_data("last trades", last_trades)
            print("Last price changes:")
            for trade in last_trades:
                print(f"Price: {trade[0]}, Volume: {trade[1]}, Timestamp: {trade[2]}, order type {trade[3]}")
        else:
            print("Failed to retrieve last price changes.")

        # Get latest price
        latest_price = self.trading_data.get_latest_price()

        if latest_price is not None:
            self.world_instance.set_data("last price", latest_price)
            print(f"The latest price for {self.trading_data.trading_pair} is: {latest_price}")

        # Get ticker
        ticker = self.trading_data.get_ticker()

        if ticker is not None:
            self.world_instance.set_data("ticker", ticker)


#data creation

# main_instance = Main()
# dataCreation = dataCreation(main_instance.world_instance, trading_pair="SOLEUR")
# for _ in range(100):
#     main_instance.run_program()
#     data = dataCreation.collect_process_data()
#     dataCreation.save(data, trading_pair="SOLEUR")

#train model

if __name__ == "__main__":
    main_instance = Main()

    file_path =r'C:\Users\Filip\PycharmProjects\tensorEnv\crypto\data\combined_data_{}.json'.format( main_instance.trading_pair)

    test_path =r'C:\Users\Filip\PycharmProjects\tensorEnv\crypto\data\test_json'



    # Create an instance of the CryptoPricePredictor class
    crypto_predictor = CryptoPricePredictor(file_path, main_instance.world_instance, test_path)

    # Train LSTM models
    crypto_predictor.train_lstm_models()

    predictions = crypto_predictor.predict_lstm_target()
    print("predictions", predictions)



















