import keras
from make_slice import make_slice, make_slice_back
from scraper import Scraper
from indicators import count_indicators
from neural import NeuralRNN
from visualizer import Visualizer
from read_settings import get_settings


def main():
    settings = get_settings('settings.json')

    symbol = settings['symbol']
    train_coeff = settings['train_coeff']
    window_size = settings['window_size']
    batch_size = settings['batch_size']
    pred_start_date = settings['pred_start_date']
    next_day_date = settings['next_day_date']
    days = settings['days']
    epoches = settings['epoches']

    start_date = "2006-12-31"
    end_date = "2023-08-10"
    scrapper = Scraper(symbol, start_date, end_date)
    history_data = scrapper.parse_quotes()
    indicators = count_indicators(history_data)

    nrnn = NeuralRNN(indicators)
    nrnn.df = nrnn.create_samples(nrnn.indicators)

    train_size = int(nrnn.df.shape[0] * train_coeff)
    train_df = nrnn.df.iloc[:train_size]
    val_df = nrnn.df.iloc[train_size:]

    nrnn.scaler.fit(nrnn.df[["rsi", "cci", "adx", "macd"]])

    train_ds = nrnn.make_dataset(
        df=train_df,
        window_size=window_size,
        batch_size=batch_size,
        use_scaler=True,
        shuffle=True,
    )
    val_ds = nrnn.make_dataset(
        df=val_df,
        window_size=window_size,
        batch_size=batch_size,
        use_scaler=True,
        shuffle=True,
    )

    # nrnn.compile_and_fit(train_ds, val_ds, epoches)
    # nrnn.model.save("tetsrnn5000v230window32batch.keras")

    nrnn.model = keras.models.load_model("tetsrnn5000v230window32batch.keras")

    test_df = make_slice_back(history_data, pred_start_date, window_size * 2 + 66)
    real_raw_ind = make_slice(history_data, next_day_date, days)
    real_ind = count_indicators(real_raw_ind)
    real = nrnn.create_samples(real_ind)
    indicators = count_indicators(test_df)

    coeff_predictions, real = nrnn.make_predictions(
        batch_size=batch_size, window_size=window_size, indicators=indicators, real=real
    )

    visualizer = Visualizer()
    visualizer.plot_results(real, coeff_predictions)


if __name__ == "__main__":
    main()
