import pandas as pd


def get_real_data():
    data = pd.read_csv("winequality-red.csv")
    data['category'] = data['quality'] >= 7
    data.head()
    X = data[data.columns[0:11]].values
    y = data['category'].values.astype('int')
    return X, y
