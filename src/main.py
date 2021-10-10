"""
this is the main pipeline of the code
"""
from common import utilities as ut
from model import train_test_spilt
import numpy as np


def main():
    housing_data = ut.load_data(r'../Data/housing/housing.csv')
    print(housing_data.shape)
    train_data, val_data, test_data = train_test_spilt.stratified_spilt(
        housing_data,
        "median_income",
        [0., 1.5, 3.0, 4.5, 6, np.inf],
        [1, 2, 3, 4, 5],
        0.2,
        0.1,
        True
    )
    print(test_data.shape)


if __name__ == '__main__':
    main()