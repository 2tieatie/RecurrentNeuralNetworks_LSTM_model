import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.src.callbacks import Callback
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.optimizers import Adam
from keras.preprocessing import timeseries_dataset_from_array
from sklearn.preprocessing import StandardScaler
import pandas as pd


class NeuralRNN:
    def __init__(self, indicators):
        self.indicators = pd.DataFrame(indicators).dropna()
        self.model = Sequential()
        self.x_train = []
        self.y_train = []
        self.scaler = StandardScaler()
        self.model = Sequential(
            [LSTM(32, return_sequences=False), keras.layers.Dropout(0.2), Dense(5)]
        )

    def make_dataset(self, df, window_size, batch_size, use_scaler=True, shuffle=True):
        features = df[["rsi", "cci", "adx", "macd"]].iloc[:-window_size]
        if use_scaler:
            features = self.scaler.transform(features)
        data = np.array(features, dtype=np.float32)
        targets = df[["rsi", "cci", "adx", "macd", "coefficient"]].iloc[window_size:]
        ds = timeseries_dataset_from_array(
            data=data,
            targets=targets,
            sequence_length=window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        return ds

    def compile_and_fit(self, train_ds, val_ds, num_epoches: int = 20):
        self.model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(),
            metrics=[MeanAbsoluteError()],
        )
        stop_on_loss = StopTrainingOnOverfit()
        history = self.model.fit(
            train_ds,
            epochs=num_epoches,
            validation_data=val_ds,
            verbose=1,
            callbacks=[stop_on_loss],
        )
        return history

    @staticmethod
    def create_samples(indicators):
        coefficient = indicators.get("coefficient")
        ind_data = list(
            zip(
                indicators.get("rsi"),
                indicators.get("cci"),
                indicators.get("adx"),
                indicators.get("macd"),
            )
        )
        df = pd.DataFrame(ind_data, columns=["rsi", "cci", "adx", "macd"])
        df["coefficient"] = coefficient
        return df

    def make_predictions(self, window_size, batch_size, indicators, real):
        self.indicators = pd.DataFrame(indicators).dropna()
        print(self.indicators)
        self.df = self.create_samples(self.indicators).iloc[33::]
        print(self.df)
        preds = []
        for _ in range(real.shape[0]):
            test_ds = self.make_dataset(
                df=self.df,
                window_size=window_size,
                batch_size=batch_size,
                use_scaler=True,
                shuffle=False,
            )
            new_date = self.df.index[-1] + 1
            prediction = self.model.predict(test_ds)
            self.df.loc[new_date] = prediction[0]
            self.df = self.df.iloc[1:]
            preds.append(prediction[0])

        predictions = np.array(preds)
        return [row[-1] for row in predictions], real


class StopTrainingOnOverfit(Callback):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.loss_checker = None

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if self.i % 30 == 0 or self.i == 0:
            self.loss_checker = val_loss

        if loss and val_loss and val_loss > loss and val_loss > self.loss_checker * 1.1:
            self.i += 1
        else:
            self.i = 0

        if self.i == 10:
            self.model.stop_training = True
