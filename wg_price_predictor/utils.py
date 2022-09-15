import numpy as np
import pandas as pd
import os

from config.config import ROOT_DIR
from wg_price_predictor.params import dict_city_number_wggesucht
from wg_price_predictor.string_utils import standardize_characters, capitalize_city_name, german_characters, simplify_address


def create_dir(path):
    # Check whether the specified path exists or not
    if os.path.exists(path):
        pass
    else:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"The new directory has been created!")

def get_file(file_name, local_file_path='data/berlin'):
    """
    Method to get data (or a portion of it) from local environment and return a dataframe
    """
    # try:
    local_path = f'{ROOT_DIR}/{local_file_path}/{file_name}'
    df = pd.read_csv(local_path)
    print(f'===> Loaded {file_name} locally')
    return df

def save_file(df, file_name, local_file_path='housing_crawler/data/berlin'):
    # Create directory for saving
    create_dir(path = f'{ROOT_DIR}/{local_file_path}')

    # Save locally
    local_path = f'{ROOT_DIR}/{local_file_path}/{file_name}'
    df.to_csv(local_path, index=False)
    print(f"===> {file_name} saved locally")


def return_significative_coef(model):
    """
    Returns p_value, lower and upper bound coefficients
    from a statsmodels object.
    """
    # Extract p_values
    p_values = model.pvalues.reset_index()
    p_values.columns = ['variable', 'p_value']

    # Extract coef_int
    coef = model.params.reset_index()
    coef.columns = ['variable', 'coef']
    return p_values.merge(coef,
                          on='variable')\
                   .query("p_value<0.05").sort_values(by='coef',
                                                      ascending=False)

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == "__main__":

    report_best_scores()
