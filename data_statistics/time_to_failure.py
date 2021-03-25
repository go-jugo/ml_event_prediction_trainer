"""
Module to calculate failure statistics
"""
import dask
import dask.dataframe as dd
import pandas as pd

writer1 = pd.ExcelWriter('../statistics/tbf_and_ft_stats.xlsx', engine='xlsxwriter')

def calc_tbf_and_ft_statistics(df, one_hot_encoded=True, error_codes=[35, 1306, 1102, 1238, 133,
                                                                      374, 730]):
    """
    Parameters
    ----------
    df : Dask dataframe
    one_hot_encoded : Bool
        True, if error code columns in df are one hot encoded
    error_codes : List
        Error Codes for wich we want to create the statistics
    """
    cols = [col for col in df.columns if 'errorCode' in col]
    if one_hot_encoded == False:
        df = one_hot_encode_errors(df[cols]) 
    cols2keep = []
    for col in df.columns:
        for code in error_codes:
            check_string ='.' + str(code) + '.'
            if check_string in col:
                cols2keep.append(col)
    df = df[cols2keep]
    df = create_failure_period_ids(df)
    print(df.columns)
    period_cols = [col for col in df.columns if 'period_id' in col]
    #time between failure and failure time stats
    lazy_results = []
    for period_col in period_cols:
        error_code = '.' + period_col.split('.')[-3] + '.'    # important for correct match of period col and corresponding dummy col
        error_code_col = [col for col in df.columns if error_code in col and 'errorCode' in col and 'period_id' not in col][0]
        print(period_col, error_code_col)
        tmp_df = df[[period_col, error_code_col]].reset_index()
        lazy_result = dask.delayed(calc_stat)(tmp_df, error_code_col, period_col)
        lazy_results.append(lazy_result)
    lazy_results = dask.compute(*lazy_results)
    tbf_stats = [item for sublist in lazy_results for item in sublist][::2]
    ft_stats = [item for sublist in lazy_results for item in sublist][1::2]
    tbf_stats = pd.concat(tbf_stats, axis=1).astype(str).T
    ft_stats = pd.concat(ft_stats, axis=1).astype(str).T
    tbf_stats.to_excel(writer1, sheet_name= 'time_between_failure')
    ft_stats.to_excel(writer1, sheet_name= 'failure_time')
    writer1.save()
    #number days occurence stats
    dummy_cols = [col for col in df.columns if 'errorCode' in col and 'period_id' not in col]
    print(df.columns)
    lazy_results = []
    for col in dummy_cols:
        error_code = col.split('.')[-2]             #fetch the error code for the column name
        series = df[col]
        lazy_result = dask.delayed(calc_number_days_occurence)(series, error_code)
        lazy_results.append(lazy_result)
    lazy_results = dask.compute(*lazy_results)
    result = pd.concat(lazy_results)
    result.to_excel('../statistics/number_days_occurence.xlsx')
    
def calc_number_days_occurence(series, error_code):
    number_days_occurence = series.groupby([series.index.year, series.index.month,
                                            series.index.day]).max().sum()
    number_months_occurence = series.groupby([series.index.year, series.index.month]).max().sum()
    result = pd.DataFrame({'error_code': [error_code],
              'number_days_occurence': [number_days_occurence],
              'number_months_occurence': [number_months_occurence]})
    return result
    
def calc_stat(tmp_df, error_code_col, period_col):
    tbf_and_ft = tmp_df.groupby([error_code_col, period_col])['global_timestamp'].agg({calc_len_period}).reset_index()
    #drop result of last period as it is censored (i.e no error at the end)
    tbf_and_ft = tbf_and_ft[tbf_and_ft[period_col] != tbf_and_ft[period_col].max()]
    new_name = error_code_col[:-2] + '.' + 'tbf'
    new_name2 = error_code_col[:-2] + '.' + 'ft'
    tbf = tbf_and_ft[tbf_and_ft[error_code_col] == 0].rename(columns = {'calc_len_period':new_name})    #0:error does not occur
    tbf_stat = tbf[new_name].describe()        
    ft = tbf_and_ft[tbf_and_ft[error_code_col] == 1].rename(columns = {'calc_len_period':new_name2})    #1:error occurs
    ft_stat = ft[new_name2].describe()
    return tbf_stat, ft_stat

def calc_len_period(series):
    series_list = series.tolist()
    return series_list[-1] - series_list[0]

def create_failure_period_ids(df):
    """
    Parameters:
    ----------
    df : Dask dataframe with one hot encoded error columns
    
    Returns:
    -------
    df : Dask dataframe
       For each error col a period id col is created
       New col starts with 1 and is incremented each time the error occurs or no longer occurs.
    """     
    error_cols = [col for col in df.columns if 'errorCode' in col]
    npartitions = df.npartitions
    lazy_results = []
    for col in error_cols:
        series = df[col]
        new_name = str(series.name) + '.period_id'
        lazy_result = dask.delayed(calc_periodId_series)(series, new_name, npartitions)
        lazy_results.append(lazy_result)
    lazy_results = dask.compute(*lazy_results)
    for series in lazy_results:
        name = series.name
        df[name] = series
    return df

def calc_periodId_series(series, new_name, npartitions):
    period_id = (series != series.shift()).cumsum()
    period_id.name = new_name
    dask_series = dd.from_pandas(period_id, npartitions=npartitions)
    return dask_series

def one_hot_encode_errors(df):
    errorcode_column_list = [col for col in df.columns if 'errorCode' in col]
    error_df = df[errorcode_column_list].astype(float).astype(str)
    error_df = error_df.categorize(columns=errorcode_column_list)
    error_df = dd.get_dummies(error_df, prefix_sep='.', prefix=errorcode_column_list)
    cols2drop = [col for col in error_df.columns if '-2147483648.0' in col or '0.0' in col]
    error_df = error_df.drop(columns=cols2drop, errors='ignore')
    return error_df