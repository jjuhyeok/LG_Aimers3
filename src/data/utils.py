import numpy as np
import pickle


def zero_filter(df):
    df_filtered = df[(df != 0)]
    return df_filtered


def correation(x, y):
    bar_x = x.mean()
    bar_y = y.mean()
    numerator = np.sum((x - bar_x) * (y - bar_y))

    left = np.sum((x - bar_x) ** 2)
    right = np.sum((y - bar_y) ** 2)
    denominator = np.sqrt(left * right)
    return numerator / denominator


def make_wide_format(train, col_name):
    train_data = train.pivot(index='ID', columns='time', values=col_name).reset_index()
    train_data.columns = ['ID'] + [col.strftime('%Y-%m-%d') for col in train_data.columns[1:]]
    return train_data


def make_month(df):  # O
    dt = df['time'].astype('str')
    month_data = pd.to_datetime(dt)
    md = month_data.dt.month
    df['month'] = md
    return df


def group_season(df):  # O
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5), 'season'] = '봄'
    df.loc[(df['month'] == 6) | (df['month'] == 7) | (df['month'] == 8), 'season'] = '여름'
    df.loc[(df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11), 'season'] = '가을'
    df.loc[(df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2), 'season'] = '겨울'
    return df['season']


def make_week(df):  # V if null
    dt = df['time'].astype('str')
    data = pd.to_datetime(dt)
    week = [i.weekday() for i in data]
    df['week'] = week
    df.loc[(df['week'] <= 4), 'week'] = 0
    df.loc[(df['week'] > 4), 'week'] = 1
    return df['week']


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
