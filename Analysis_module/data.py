from typing import Dict
import pandas as pd
import numpy as np

class Data:
  def __init__(self) -> None:
    print('hi from data')

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