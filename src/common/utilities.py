"""
this file contains utility functions
"""
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    this function loads and return the data
    :param path: path of the data file
    :return: dataframe of the data file
    """
    # test commit
    data = pd.read_csv(path)
    return data


