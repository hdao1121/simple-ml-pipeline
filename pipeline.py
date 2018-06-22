import os
import tarfile
from six.moves import urllib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()

def explore_data(data):
    print(data.head())
    print(data.info())
    print(data["ocean_proximity"].value_counts())
    print(data.describe())

def visualize_data(data):
    data.hist(bins=50, figsize=(20,15))
    plt.show()

def split_data(data):
    data["income_cat"] = np.ceil(data["median_income"]/1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)

    # check each category percentage
    print(data["income_cat"].value_counts() / len(data))

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]


split_data(housing)