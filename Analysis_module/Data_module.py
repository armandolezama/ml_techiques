from typing import Dict
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import pandas as pd

class Data_module:
  def __init__(self) -> None:
    self.load_data()
    self.apple_quality_data = pd.read_csv('./Data/apple_quality.csv')
    self.student_data = pd.read_csv('./Data/student_data.csv',header=0, sep=';')

  def load_data(self):
     self.boston_housing_x, self.boston_housing_y = load_boston(return_X_y=True)
     self.iris_x, self.iris_y = load_iris(return_X_y=True)
     self.minist_x, self.minist_y = fetch_openml('mnist_784', version=1, return_X_y=True)
     self.mall_customer_x = self.load_mall_customer()
     self.breast_cancer_x, self.breast_cancer_y = load_breast_cancer(return_X_y=True)
     self.digits_x, self.digits_y = load_digits(return_X_y=True)

  def load_mall_customer(self):
    return pd.read_csv('./Data/Mall_Customers.csv')

  def make_bootstraps(self, data: np.array, n_bootstraps: int=100) -> Dict[str, Dict[str, np.array]]:

    """
    Function to generate bootstrapped samples

    Inputs:
        data         -> array of input data
        n_bootstraps -> integer number of bootstraps to produce

    Outputs:
        {'boot_n': {'boot': np.array, 'test': np.array}} -> dictionary of dictionaries containing
                                                            the bootstrap samples & out-of-bag test sets
    """

    # initialize output dictionary, unique value count, sample size, & list of indices
    dc       = {}
    n_unival = 0
    sample_size = data.shape[0]
    idx = [i for i in range(sample_size)]
    # loop through the required number of bootstraps
    for b in range(n_bootstraps):
        # obtain boostrap samples with replacement
        sidx = np.random.choice(idx,replace=True,size=sample_size)
        sidx =data.index[sidx]
        b_samp = data.loc[sidx]
        # compute number of unique values contained in the bootstrap sample
        n_unival += len(set(sidx))
        # obtain out-of-bag samples for the current b
        oob_idx = list(set(idx) - set(sidx))
        t_samp = np.array([])
        if oob_idx:
            t_samp = data.iloc[oob_idx, :]
        # store results
        dc['boot_'+str(b)] = {'boot':b_samp,'test':t_samp}
    # state the mean number of unique values in the bootstraps
    print('Mean number of unique values in each bootstrap: {:.2f}'.format(n_unival/n_bootstraps))
    # return the bootstrap results
    return(dc)
