import pandas as pd
import dask.dataframe as dd
from ..monitoring.time_it import timing

@timing
def remove_error_codes(df, dependent_variable, skip=True):
    if not skip:
        errorCode_columns = [col for col in df.columns if 'errorCode' in col]
        errorCode_columns.remove(dependent_variable)
        df = df.drop(columns=errorCode_columns)
    return df