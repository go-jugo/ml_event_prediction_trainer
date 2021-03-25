import pandas as pd
from ..monitoring.time_it import timing
import dask
import dask.dataframe as dd
import re

def value_counter(series):
    return pd.Series(series.value_counts(dropna=False), name=series.name)

@timing
def calculate_statistics(df):

    errorcode_column_list = [col for col in df.columns if 'errorCode' in col]
    df_result = pd.DataFrame()

    lazy_results = []
    for errorcode in errorcode_column_list:
        series = df[errorcode]
        lazy_result = dask.delayed(value_counter)(series)
        lazy_results.append(lazy_result)
    lazy_results = dask.compute(*lazy_results)
    for result in lazy_results:
        df_result = pd.concat([df_result, result], ignore_index=True, axis=1)
    df_result.columns = errorcode_column_list
    df_result.to_excel('../statistics/errorcode.xlsx')

@timing
def calculate_statistics_by_year(df):

    errorcode_column_list = [col for col in df.columns if 'errorCode' in col]
    df_result = pd.DataFrame()

    lazy_results = []
    for errorcode in errorcode_column_list:
        series = df[errorcode]
        lazy_result = dask.delayed(value_counter)(series)
        lazy_results.append(lazy_result)
    lazy_results = dask.compute(*lazy_results)
    for result in lazy_results:
        df_result = pd.concat([df_result, result], axis=0)
    df_result.columns = ['errorCode']
    df_result = df_result.groupby(level=0).sum()
    df_result.to_excel('../statistics/errorcode_statistics_by_year.xlsx')
    
def cumsum_since_occurence(series):
    name = series.name
    series = series.reset_index()
    series['timedelta'] = (series['global_timestamp'] - series['global_timestamp'].shift(1)).astype('timedelta64[s]')
    series = series.set_index('global_timestamp')
    mask = (series[name] == 1).cumsum().shift(1)
    series = series.drop(columns=name)
    time_col =  str(name[:-2]) + '.timeSinceOccurence'
    series[time_col] = series.groupby(mask.where(mask > 0)).timedelta.cumsum()
    return series[time_col]

def calc_time_since_occurence_statistics(df, relevant_col = 'components.cont.conditions.logic.errorCode'):
    df_error_col = df.categorize(columns=[relevant_col])[relevant_col]
    df_dummy = dd.get_dummies(df_error_col, prefix_sep='.', prefix=relevant_col)
    lazy_results = []
    for col in df_dummy.columns:
        series = df_dummy[col]
        lazy_result = dask.delayed(cumsum_since_occurence)(series)
        lazy_results.append(lazy_result)
    series_collection = dask.compute(*lazy_results)
    excel_writer = pd.ExcelWriter('../statistics/time_since_occurence_per_error_code.xlsx', engine='xlsxwriter')
    for series in series_collection:
        descriptive_stat = series.describe()
        descriptive_stat.to_excel(excel_writer, sheet_name= str('_'.join(re.split('\.(?!\$)|\_(?!.*\.)', series.name)[-3:-1])))
    excel_writer.save() 