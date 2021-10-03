"""
this is the main pipeline of the code
"""
from common import utilities as ut


def main():
    housing_data = ut.load_data(r'C:\Users\WarnaK\OneDrive - John Keells Holdings PLC\workspace\python_train\hands_on_ml\hands_on_ml\Data\housing\housing.csv')
    print(housing_data.shape)


if __name__ == '__main__':
    main()