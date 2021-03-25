import pandas as pd
import dask.dataframe as dd
import dask

def calc_data_acquisition_statistics(df):
    start = df.head(1).index
    end = df.tail(1).index
    length = (end - start)
    num_logs = len(df)
    num_variables = len(df.columns)
    num_numerical_variables = len([col for col in df.columns if 'samples' in col])
    num_categorial_variables = num_variables - num_numerical_variables
    number_distinct_errors = calc_number_distinct_errors(df)
    
    stats = {
    'Observation period start' : [start],
    'Observation period end' : [end],
    'Length observation period' : [length],
    'Number machine logs' : [num_logs],
    'Number variables' : [num_variables],
    'Number numerical variables' : [num_numerical_variables],
    'Number categorial variables' : [num_categorial_variables],
    'Number distinct error codes' : [number_distinct_errors]
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.to_excel('../statistics/data_acquisition_statistics.xlsx')
    sampling_frequency_stat = calc_sampling_statistic(df)
    sampling_frequency_stat.to_excel('../statistics/sampling_frequency_statistics.xlsx')
      
def calc_number_distinct_errors(df):
    error_cols = [col for col in df.columns if 'errorCode' in col]
    lazy_results = []
    for col in error_cols:
        series = df[col]
        lazy_result = dask.delayed(calc_unique)(series)
        lazy_results.append(lazy_result) 
    collection = dask.compute(*lazy_results)
    collection = [item for sublist in collection for item in sublist]     #flatten list
    collection = [item for item in collection if str(item) != 'nan']      #remove nan
    number_distinct_errors = len(list(dict.fromkeys(collection))) - 1     #do not count ok code
    return number_distinct_errors

def calc_unique(series):
    return series.drop_duplicates().tolist()

def calc_sampling_statistic(df):
    stat = df.index.compute().to_series(name='sampling frequency').diff().reset_index().describe().astype(str)
    return stat