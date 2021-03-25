from .config import v_dask, logging_level, logging_color
from .logger import init_logger, get_logger
# import pandas as pd
# import numpy as np
# from .data_merge import create_global_dataframe
from .data_cleansing import impute_missing_values, value_based_column_filter, one_hot_encode_categories, remove_correlated_features, remove_error_codes
from .feature_engineering import standardize_features, generate_error_code_indicators
from .feature_engineering.extract_windows_and_engineer_features_with_tsfresh import extract_windows_and_features
from .data_cleansing.adjust_sampling_frequency import adjust_sampling_frequency
from .ml_evaluation.eval import eval
from .ml_evaluation.get_x_y import get_x_y
from .feature_engineering.remove_global_timestamp import remove_global_timestamp
# from multiprocessing import Pool
from .feature_engineering.create_error_code_col import create_error_code_col
from types import SimpleNamespace
from dask import dataframe as dd
import pickle
import openpyxl

init_logger(logging_level, logging_color)

logger = get_logger(__name__.split(".", 1)[-1])


def df_from_csv(csv_path: str, time_col: str, sorted=False):
    df: dd.DataFrame = dd.read_csv(csv_path)
    df[time_col] = dd.to_datetime(df[time_col])
    df = df.set_index(time_col, sorted=sorted)
    return df


def clf_from_pickle_file(pickle_file_path: str):
    with open(pickle_file_path, 'rb') as file:
        return pickle.load(file)


def clf_from_pickle_bytes(pickle_bytes: bytes):
    return pickle.loads(pickle_bytes)


def run_pipeline(df, config):
    c = SimpleNamespace(**config)
    logger.debug('Data cleansing')
    df = value_based_column_filter.value_filter(df, v_dask=v_dask)
    error_code_series = create_error_code_col(df, c.target_col)
    df = adjust_sampling_frequency(df, sampling_frequency=c.sampling_frequency, v_dask=v_dask)
    df = impute_missing_values.impute_missing_values(df, v_dask=v_dask, string_imputation_method=c.imputations_technique_str,
                                                     numeric_imputation_method=c.imputation_technique_num)
    df = impute_missing_values.slice_valid_data(df, v_dask=v_dask)
    df = one_hot_encode_categories.one_hot_encode_categories(df, errorcode_col=c.target_col, v_dask=v_dask)
    df = remove_error_codes.remove_error_codes(df, dependent_variable=c.target_col)
    logger.debug('Feature engineering')
    df = extract_windows_and_features(df, error_code_series, errorcode_col=c.target_col,
                                      errorcode=c.target_errorCode, window_length=c.ts_fresh_window_length,
                                      window_end=c.ts_fresh_window_end, balance_ratio=c.balance_ratio,
                                      minimal_features=c.ts_fresh_minimal_features, v_dask=v_dask)
    logger.debug('ML evaluation')
    df = standardize_features.standardize_features(df, errorcode_col=c.target_col, scaler=c.scaler)
    df = remove_global_timestamp(df)
    df, y = get_x_y(df, target_variables_column=c.target_col, errorcode=c.target_errorCode)
    ml_model = eval(df, y, config=config, crossvalidation=c.cv, clf=c.ml_algorithm, random_state=c.random_state)
    return ml_model