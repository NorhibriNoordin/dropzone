import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler


def process_data():
    training_data_df = pd.read_csv("filepath.csv")
    test_data_df = pd.read_csv("filepath.csv")

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_training = scaler.fit_transform(training_data_df)
    scaled_testing = scaler.transform(test_data_df)

    print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(
        scaler.scale_[8], scaler.min_[8]))

    scaled_training_df = pd.DataFrame(
        scaled_training, columns=training_data_df.columns.values)
    scaled_testing_df = pd.DataFrame(
        scaled_testing, columns=test_data_df.columns.values)

    scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
    scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)


def create_model():

    training_df = pd.read_csv("filepath_scaled.csv")

    X = training_df.drop('model_id', axis=1).values

    Y = training_df[['model_id']].values

    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(
        X,
        Y,
        epochs=50,
        shuffle=True,
        verbose=2
    )


def train_model():
    training_data_df = pd.read_csv("filepath_scaled.csv")

    X = training_data_df.drop('total_earnings', axis=1).values
    Y = training_data_df[['total_earnings']].values

    model = load_model("model_recommender.h5")

    model.fit(
        X,
        Y,
        epochs=50,
        shuffle=True,
        verbose=2
    )
