import os
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"

class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)

def fetch_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
# fetch_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()

def explore_data(data):
    print(data.head())
    print(data.info())
    print(data["ocean_proximity"].value_counts())
    print(data.describe())

def split_data(data):
    data["income_cat"] = np.ceil(data["median_income"]/1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)

    # check each category percentage
    # print(data["income_cat"].value_counts() / len(data))

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    for set in (strat_test_set, strat_train_set):
        set.drop(["income_cat"], axis=1, inplace=True)
    return strat_train_set, strat_test_set

train_set, test_set = split_data(housing)

def visualize_data(data):
    data.hist(bins=50,figsize=(20,15))
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
              s=housing["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    # print(data.corr())
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12,8))
    plt.show()
# visualize_data(train_set)

# prepare data
train_set_labels = train_set["median_house_value"].copy()
train_set = train_set.drop("median_house_value", axis=1)

# drop 'ocean_proximity' which is a text attribute
train_set_num = train_set.drop("ocean_proximity", axis=1)

# data cleaning- fills in missing fields with an average value for text columns
num_attribs = list(train_set_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', CustomLabelBinarizer()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

train_set_prep = full_pipeline.fit_transform(train_set)

#Start training data
lin_reg = LinearRegression()
lin_reg.fit(train_set_prep, train_set_labels)

some_data = train_set.iloc[:10000]
some_labels = train_set_labels.iloc[:10000]
some_data_prepared = full_pipeline.transform(some_data)
print(some_data_prepared.shape)

print(lin_reg.predict(some_data_prepared))
