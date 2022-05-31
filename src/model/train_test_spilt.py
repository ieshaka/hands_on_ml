"""
this file contains the fuctions to break the dataset into train,test and validation
"""
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_spilt(
        data: pd.DataFrame,
        stra_col: str,
        bins: list,
        labels: list,
        test_size: float,
        val_size: float,
        is_numerical: bool
):
    """
    This functions breaks the dataset into train test and validation based on
    the stra_col and the test and val sizes
    :param data: dataframe of the dataset
    :param stra_col: column name that the strata is created on
    :param bins: list of bins if the strata column is continues
    :param labels: list of labels for the bins
    :param test_size: size of the test set
    :param val_size: size of the train set
    :param is_numerical: if the stratifying column is continues
    :return:
    """
    if is_numerical:
        data['strata_cat'] = pd.cut(data[stra_col], bins=bins, labels=labels)
    else:
        data['strata_cat'] = data[stra_col]

    test_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    for train_index, test_index in test_split.split(data, data['strata_cat']):
        start_train_set = data.loc[train_index]
        start_test_set = data.loc[test_index]

    val_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)

    start_train_set_reindex = start_train_set.reset_index().drop('index', axis=1)

    for train_index, val_index in val_split.split(start_train_set_reindex, start_train_set_reindex['strata_cat']):
        start_train_set = start_train_set_reindex.loc[train_index]
        start_val_set = start_train_set_reindex.loc[val_index]

    for data_ in (start_train_set, start_val_set, start_test_set):
        data_.drop('strata_cat', axis=1, inplace=True)

    return start_train_set, start_val_set, start_test_set
