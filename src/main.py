"""
this is the main pipeline of the code
"""
from common import utilities as ut
from model import train_test_spilt
import numpy as np
from model.transformations import CombinedAttributesAdder
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def main():
    housing_data = ut.load_data(r'./Data/housing/housing.csv')
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
    
    housing = pd.concat([train_data,val_data])  

    l_cols_numeric = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']

    l_cols_categorical = ['ocean_proximity']

    num_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy="median")),
            ('attrib_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ('num', num_pipeline, l_cols_numeric),
            ('cat', OneHotEncoder(), l_cols_categorical)
        ]
    )

    housing_prepared = full_pipeline.fit_transform(housing)

    print(housing_prepared.shape)



if __name__ == '__main__':
    main()