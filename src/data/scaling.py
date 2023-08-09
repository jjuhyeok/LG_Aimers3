from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd
import json


def scale(train_data_origin):
    numeric_cols = train_data_origin.columns[1:]
    min_values = train_data_origin[numeric_cols].min(axis=1)
    max_values = train_data_origin[numeric_cols].max(axis=1)
    ranges = max_values - min_values
    ranges[ranges == 0] = 1
    train_data_origin[numeric_cols] = (train_data_origin[numeric_cols].subtract(min_values, axis=0)).div(ranges, axis=0)
    scale_min_dict = min_values.to_dict()
    scale_max_dict = max_values.to_dict()
    return train_data_origin, scale_min_dict, scale_max_dict


def dict2json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f)


def encoding(data):
    label_encoder = LabelEncoder()
    categorical_columns = ['대분류', '중분류', '소분류', '브랜드']
    for col in categorical_columns:
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])


if __name__ == "__main__":
    train_data = pd.read_csv('../data/raw/train.csv').drop(columns=['ID', '제품'])
    sales = pd.read_csv('../data/raw/sales.csv').drop(columns=['ID', '제품'])
