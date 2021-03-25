import pandas as pd
import dask.dataframe as dd
import dask as d 
import numpy as np
from ..monitoring.time_it import timing
from ..tools.series_list_to_df import series_list_to_df
from ..logger import get_logger

logger = get_logger(__name__.split(".", 1)[-1])

artifical_timestamp = 'global_timestamp'

def impute_series(pandas_series, imputation_method, limit):
     imputed_series = pandas_series.interpolate(method = imputation_method, limit = limit)
     return imputed_series
   

def compute_first_and_last(series):
    first_and_last = [series.first_valid_index(), series.last_valid_index()]
    return first_and_last


def check_input(string_imputation_method, numeric_imputation_method):
    valid_string_interpolation = ['pad']
    if string_imputation_method not in valid_string_interpolation:
        raise ValueError('"{}" is not a valid imputation method for string columns'.format(string_imputation_method))  
    valid_num_interpolation = ['pad', 'linear', 'time']
    if numeric_imputation_method not in valid_num_interpolation:
        raise ValueError('"{}" is not a valid imputation method for numeric columns'.format(numeric_imputation_method)) 


@timing
def impute_missing_values(df, string_imputation_method='pad', numeric_imputation_method='pad',
                          limit=None, v_dask=True):
    check_input(string_imputation_method, numeric_imputation_method)
    columns = list(df.columns)
    numeric_cols = list(df.select_dtypes(include=['number', 'datetime']).columns)
    string_cols = list(df.select_dtypes(exclude=['number', 'datetime']).columns)

    lazy_results = []
    for col in columns:
        series = df[col]
        if col in string_cols:
             lazy_result = d.delayed(impute_series)(series, imputation_method=string_imputation_method, limit=limit)
        if col in numeric_cols:
             lazy_result = d.delayed(impute_series)(series, imputation_method=numeric_imputation_method, limit=limit)
        lazy_results.append(lazy_result) 
    series_collection = d.compute(*lazy_results)

    df = series_list_to_df(series_collection, v_dask)
    return df

@timing
def slice_valid_data(df, extrapolate=True,  v_dask=True):
    if v_dask:
        lazy_results = []       
        for col in df.columns:
            series = df[col]
            lazy_result = d.delayed(compute_first_and_last)(series)
            lazy_results.append(lazy_result) 
        lazy_results = d.compute(*lazy_results)
        first_indices = []
        last_indices = []
        for result in lazy_results:
            first_indices.append(result[0])
            last_indices.append(result[1])  
        first_indices_clean = [index for index in first_indices if index != None]
        last_indices_clean = [index for index in last_indices if index != None]
        first_index_overall = max(first_indices_clean)
        last_index_overall = min(last_indices_clean)
        logger.debug('First valid index: ' + str(first_index_overall))
        logger.debug('Last valid index ' + str(last_index_overall))
        if not extrapolate:
            df_reduced = df.loc[first_index_overall:last_index_overall]
        else:
            df_reduced = df.loc[first_index_overall:]
        return df_reduced

    if not v_dask:
        first_valid_loc = df.apply(lambda col: col.first_valid_index()).max()
        last_valid_loc = df.apply(lambda col: col.last_valid_index()).min()
        logger.debug(first_valid_loc)
        logger.debug(last_valid_loc)
        if not extrapolate:
            df_reduced = df.loc[first_valid_loc:last_valid_loc,:]
        else:
            df_reduced = df.loc[first_valid_loc:,:]
        return df_reduced

