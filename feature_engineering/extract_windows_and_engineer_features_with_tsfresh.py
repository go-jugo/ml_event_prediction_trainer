import datetime
import pandas as pd
import dask.dataframe as dd
from sklearn.utils import shuffle
import dask
import glob
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from ..monitoring.time_it import timing
from math import ceil
import copy
import random
from ..logger import get_logger
from ..config import debug_mode
from datetime import timezone

logger = get_logger(__name__.split(".", 1)[-1])


def calculate_window(df, window_start_date, window_end_date, element, minimal_features, window_length, errorcode_col,
                     extract_negative_examples=False):

    df_window = df.loc[window_start_date:window_end_date].copy()

    global_timestamp = element[0]
    errorcode_value = element[1]

    df_window['id'] = 1

    if df_window.empty:
        logger.debug('Empty')
        if extract_negative_examples:
            return pd.DataFrame(data={'global_timestamp': [global_timestamp]})
        else:
            return

    df_window = df_window.fillna(0)

    if minimal_features:
        extracted_features = extract_features(df_window, column_id='id', n_jobs=0,
                                              default_fc_parameters=MinimalFCParameters(), disable_progressbar=not debug_mode)
    else:
        extracted_features = extract_features(df_window, column_id='id', column_sort='global_timestamp', n_jobs=0,
                                              default_fc_parameters=EfficientFCParameters(), disable_progressbar=not debug_mode)

    extracted_features['global_timestamp'] = global_timestamp
    extracted_features[errorcode_col] = errorcode_value
    extracted_features['samples_used'] = len(df_window)
    extracted_features['window_start'] = window_start_date
    extracted_features['window_end'] = window_end_date
    extracted_features['window_length'] = window_length

    return extracted_features


def get_clean_errorcode_column_to_process(df, error_code_series, errorcode_col, errorcode, window_end, window_length, negative_examples=False):

    first_index = pd.to_datetime(df.reset_index().loc[0]['time'].compute().values[0]).replace(tzinfo=timezone.utc)

    if negative_examples:
        target_errorcode_timestamp_list = list(error_code_series[error_code_series[errorcode_col] == errorcode].index.compute())
        exclude_windows = []
        for element in target_errorcode_timestamp_list:
            exclude_windows.append([element + datetime.timedelta(seconds=(window_end)),
                                    element + datetime.timedelta(seconds=(window_end + window_length))])
        df_process = error_code_series
        df_process = df_process.fillna(method='ffill')
        df_process = df_process.loc[df_process[errorcode_col] != errorcode].compute()
        df_process = df_process.loc[first_index:, ]
        df_process = df_process.dropna()
        for element in exclude_windows:
            mask = ~((df_process.index >= element[0]) & (df_process.index <= element[1]))
            df_process = df_process.loc[mask]
        return df_process
    else:
        df_process = error_code_series
        df_process = df_process.loc[df_process[errorcode_col] == errorcode].compute()
        df_process = df_process.loc[first_index: ,]
        target_errorcode_timestamp_list = [element for element in df_process.index]
        timestamps_to_process = [copy.deepcopy(target_errorcode_timestamp_list[0])]
        target_errorcode_timestamp_list.pop(0)
        for element in sorted(target_errorcode_timestamp_list):
            current_element = timestamps_to_process[-1]
            time_limit = element - datetime.timedelta(seconds=window_end + window_length)
            if current_element < time_limit:
                timestamps_to_process.append(element)
        df_process = df_process.loc[timestamps_to_process]
        return df_process


@timing
def extract_windows_and_features(df, error_code_series, errorcode_col, errorcode, window_length, window_end, balance_ratio,
                                 minimal_features = False, v_dask=True):


    df_process = get_clean_errorcode_column_to_process(df, error_code_series, errorcode_col, errorcode,
                                                       window_end, window_length, negative_examples=False)
    number_of_errors = len(df_process)

    logger.debug('Number of errorCode Features to process: ' + str(len(df_process)))

    if number_of_errors < 5:
        raise ValueError('Not enough data to generate ML model')

    number_of_non_errors = int(ceil(number_of_errors * ((1 / balance_ratio) - 1)))

    df_balance = get_clean_errorcode_column_to_process(df, error_code_series, errorcode_col, errorcode,
                                                       window_end, window_length, negative_examples=True)

    if len(df_balance) < number_of_errors:
        raise ValueError('Not enough data to generate ML model')

    df_balance = df_balance.sample(n=number_of_non_errors)
    logger.debug('Number of Default Features to process: ' + str(len(df_balance)))

    if len(df_process) > 0:
        df_process = pd.concat([df_process, df_balance])
    else:
        df_process = df_balance

    logger.debug('Number of total Features to process: ' + str(len(df_process)))
    df_process = df_process.squeeze('columns')
    process_list = list(zip(df_process.index, df_process))

    lazy_results = []
    if v_dask:
        for element in process_list:
            window_start_date = element[0] - datetime.timedelta(seconds=(window_length + window_end))
            window_end_date = element[0] - datetime.timedelta(seconds=(window_end))
            lazy_result = dask.delayed(calculate_window)(df, window_start_date, window_end_date, element,
                                                         minimal_features, window_length, errorcode_col)
            lazy_results.append(lazy_result)
    lazy_results = dask.compute(*lazy_results)
    if len(lazy_results) > 0:
        df_tsfresh = pd.concat(lazy_results)
    else:
        df_tsfresh = pd.DataFrame()
    df_tsfresh = df_tsfresh.reset_index(drop=True)
    logger.debug('Number of Features extraced: ' + str(len(df_tsfresh)))

    return df_tsfresh
