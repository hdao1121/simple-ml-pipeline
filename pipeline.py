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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
from sklearn.tree import  DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import  GridSearchCV

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

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# lin_reg = LinearRegression()
# lin_reg.fit(train_set_prep, train_set_labels)
# data_predictions = lin_reg.predict(train_set_prep)
# lin_scores = cross_val_score(lin_reg, train_set_prep, train_set_labels, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
#
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(train_set_prep, train_set_labels)
# data_predictions = tree_reg.predict(train_set_prep)
# tree_scores = cross_val_score(tree_reg, train_set_prep, train_set_labels, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-tree_scores)
#
# forest_reg = RandomForestRegressor()
# forest_reg.fit(train_set_prep, train_set_labels)
# data_predictions = tree_reg.predict(train_set_prep)
# forest_scores = cross_val_score(forest_reg, train_set_prep, train_set_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)

# display_scores(lin_rmse_scores)
# display_scores(tree_rmse_scores)
# display_scores(forest_rmse_scores)

#Fine tune model
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_set_prep, train_set_labels)

final_model = grid_search.best_estimator_
Y_test = test_set["median_house_value"]
X_test = test_set.drop("median_house_value", axis=1)
X_test_prep = full_pipeline.transform(X_test)

final_prediction = final_model.predict(X_test_prep)
final_mse = mean_squared_error(Y_test, final_prediction)
final_rmse = np.sqrt(final_mse)

print(final_rmse)