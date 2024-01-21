import argparse
import os
import tarfile
import urllib
from logging import Logger
from urllib import request as urllib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from housing.logger import configure_logger

# from urllib import request as urllib


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
imputer = SimpleImputer(strategy="median")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Download and extract the housing dataset from a given URL.

    Parameters
    ----------
    housing_url : str, optional
        The URL from which to download the housing dataset.
    housing_path : str, optional
        The local directory where the dataset will be stored.

    Returns
    -------
    None
        The function doesn't return anything. It downloads and extracts the dataset.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load the housing dataset from a CSV file.

    Parameters
    ----------
    housing_path : str, optional
        The local directory where the dataset CSV file is located.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the loaded housing dataset.

    Examples
    --------
    >>> load_housing_data()
    # Loads the housing dataset from the default location.

    >>> load_housing_data(housing_path='/path/to/custom/location')
    # Loads the housing dataset from a custom location.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    """
    Calculate the proportions of each income category in the given dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing the "income_cat" column.

    Returns
    -------
    pandas.Series
        A Series containing the proportions of each income category.

    Examples
    --------
    >>> income_cat_proportions(my_dataset)
    # Calculates the proportions of each income category in the provided dataset.
    """
    return data["income_cat"].value_counts() / len(data)


def train_test(housing):
    """
    Split the housing dataset into training and testing sets with stratified sampling based on income categories.

    Parameters
    ----------
    housing : pandas.DataFrame
        The dataset to be split.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing the stratified training and testing sets.

    Examples
    --------
    >>> train_set, test_set = train_test(my_housing_dataset)
    # Splits the housing dataset into training and testing sets using stratified sampling.
    """

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(
        housing, housing["income_cat"]
    ):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


from sklearn.base import BaseEstimator, TransformerMixin

# column index


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def preprocess(strat_train_set):
    """
    Preprocess the stratified training set for housing data.

    Parameters
    ----------
    strat_train_set : pandas.DataFrame
        The stratified training set of the housing data.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing the preprocessed housing features and labels.

    Examples
    --------
    >>> features, labels = preprocess(my_stratified_train_set)
    # Preprocesses the stratified training set of the housing data.
    """

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis=1)
    housing_cat = housing[["ocean_proximity"]]

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # numerical processing pipeline
    numerical_processor = Pipeline(
        steps=[
            (
                "imputation_mean",
                SimpleImputer(missing_values=np.nan, strategy="mean"),
            ),
            (
                "custom_transformer",
                CombinedAttributesAdder(add_bedrooms_per_room=True),
            ),
        ]
    )

    # categorical preprocessing pipeline
    categorical_processor = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    num_attributes = list(housing_num)
    cat_attributes = ["ocean_proximity"]
    extra_attributes = [
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
    ]
    preprocessor = ColumnTransformer(
        [
            ("numerical", numerical_processor, num_attributes),
            ("categorical", categorical_processor, cat_attributes),
        ]
    )

    housing_prepared = preprocessor.fit_transform(housing, housing_labels)
    cat_features_names = preprocessor.transformers_[1][
        1
    ].get_feature_names_out(cat_attributes)
    feature_names = (
        list(num_attributes) + extra_attributes + list(cat_features_names)
    )
    housing_prepared = pd.DataFrame(housing_prepared, columns=feature_names)
    return housing_prepared, housing_labels


#  to get the project's directory
def get_path():
    """
    Retrieve the path of the 'mle-training' project directory.

    Returns
    -------
    str
        The absolute path of the 'mle-training' project directory.

    Examples
    --------
    >>> project_path = get_path()
    # Retrieves the absolute path of the 'mle-training' project directory.
    """
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle-training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


# to store the trained/validated datasets
def save_preprocessed(train_X, train_y, test_X, test_y, processed):
    """
    Save preprocessed data to CSV files.

    Parameters
    ----------
    train_X : pandas.DataFrame
        Features of the training set.
    train_y : pandas.Series
        Labels of the training set.
    test_X : pandas.DataFrame
        Features of the test set.
    test_y : pandas.Series
        Labels of the test set.
    processed : str
        Path to the directory where the preprocessed files will be saved.

    Returns
    -------
    None

    Examples
    --------
    >>> save_preprocessed(train_X, train_y, test_X, test_y, '/path/to/processed')
    # Saves preprocessed data to CSV files in the specified directory.
    """
    train_X.to_csv(os.path.join(processed, "train_X.csv"), index=False)
    train_y.to_csv(os.path.join(processed, "train_y.csv"), index=False)
    test_X.to_csv(os.path.join(processed, "test_X.csv"), index=False)
    test_y.to_csv(os.path.join(processed, "test_y.csv"), index=False)
