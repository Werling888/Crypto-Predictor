import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.python.keras.models import save_model, load_model


class CryptoPricePredictor:
    def __init__(self, file_path, world_instance):
        self.file_path = file_path
        self.data = self.load_data()
        self.flat_data = self.flatten_data()
        self.world_instance = world_instance
        self.models_path = r'C:\Users\Filip\PycharmProjects\tensorEnv\crypto\model'
        np.os.makedirs(self.models_path, exist_ok=True)
        # Initialize models and scalers as dictionaries
        self.models = {}
        self.scalers = {}


    def load_data(self):
        with open(self.file_path, "r") as file:
            data = json.load(file)
        return data

    def flatten_data(self):
        flat_data = []

        for record in self.data:
            flat_record = {
                "Overall Average sell Price": record["Overall Average sell Price"],
                "Overall Total sell Volume": record["Overall Total sell Volume"],
                "Overall Average buy Price": record["Overall Average buy Price"],
                "Overall Total buy Volume": record["Overall Total buy Volume"],
                "The lowest sell Price": record["The lowest sell Price"],
                "The volume at the lowest sell price": record["The volume at the lowest sell price"],
                "The highest buy Price": record["The highest buy Price"],
                "The volume at the highest buy price": record["The volume at the highest buy price"],
                "current_price_0": record["current_price_0"],
                "current_price_1": record["current_price_1"],
                "current_price_2": record["current_price_2"],
                "current_price_3": record["current_price_3"],
                "current_price_4": record["current_price_4"],
            }
            flat_data.append(flat_record)

        return flat_data

    def train_lstm_models(self):
        features = ["Overall Average sell Price", "Overall Total sell Volume", "Overall Average buy Price",
                    "Overall Total buy Volume","The lowest sell Price","The volume at the lowest sell price","The highest buy Price","The volume at the highest buy price"]
        targets = ["current_price_1", "current_price_2", "current_price_3", "current_price_4"]

        for target in targets:
            X = pd.DataFrame(self.flat_data)[features]
            y = pd.DataFrame(self.flat_data)[target]

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

            model = Sequential()
            model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=0)

            y_pred = model.predict(X_test)

            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_test_rescaled = scaler.inverse_transform(y_test)

            # Store model and scaler in dictionaries
            self.models[target] = model
            self.scalers[target] = scaler

            mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
            print(f"Mean Squared Error for {target}: {mse}")

            # Save the trained model using TensorFlow's save_model method
            model_path = np.os.path.join(self.models_path, f'model_{target}.h5')
            save_model(model, model_path)

            # Optionally save the scaler using joblib
            scaler_path = np.os.path.join(self.models_path, f'scaler_{target}.joblib')
            joblib.dump(scaler, scaler_path)

    def load_lstm_models(self):
        features = ["Overall Average sell Price", "Overall Total sell Volume", "Overall Average buy Price",
                    "Overall Total buy Volume"]
        targets = ["current_price_15", "current_price_30", "current_price_45", "current_price_60"]

        for target in targets:
            # Load the model
            model_path = np.os.path.join(self.models_path, f'model_{target}.h5')
            model = load_model(model_path)

            # Load the scaler
            scaler_path = np.os.path.join(self.models_path, f'scaler_{target}.joblib')
            scaler = joblib.load(scaler_path)

            # Store the loaded model and scaler in dictionaries
            self.models[target] = model
            self.scalers[target] = scaler

    def predict_lstm_target(self):
        predictions = {}

        # Use the trained model and scaler for prediction
        input_features = {
            "Overall Average sell Price": 89.2459,
            "Overall Total sell Volume": 11921.348000000005,
            "Overall Average buy Price": 87.0047,
            "Overall Total buy Volume": 13822.864,
            "The lowest sell Price": 88.16,
            "The volume at the lowest sell price": 212.0,
            "The highest buy Price": 88.15,
            "The volume at the highest buy price": 2.0,
            "current_price_0": 88.12,
            "current_price_1": 88.12,
            "current_price_2": 88.12,
            "current_price_3": 88.12,
            "current_price_4": 88.12
        }

        targets = ["current_price_1", "current_price_2", "current_price_3", "current_price_4"]


        for target in targets:
            # Get the model and scaler corresponding to the target
            model = self.models[target]
            scaler = self.scalers[target]

            # Features used during training
            trained_features = ["Overall Average sell Price", "Overall Total sell Volume", "Overall Average buy Price",
                                "Overall Total buy Volume", "The lowest sell Price", "The volume at the lowest sell price", "The highest buy Price", "The volume at the highest buy price"]

            # Create an array from the input features in the correct order
            input_data = np.array([input_features[feature] for feature in trained_features]).reshape(1, 1, len(
                trained_features))

            # Check the shape of input_data
            print("Input shape:", input_data.shape)

            # Make the prediction
            y_pred_scaled = model.predict(input_data)

            # Check the shape of y_pred_scaled
            print("Prediction shape:", y_pred_scaled.shape)

            # Inverse transform the scaled prediction to get the actual prediction
            y_pred_rescaled = scaler.inverse_transform(y_pred_scaled)

            predictions[target] = float(y_pred_rescaled[0, 0])

        return predictions











