# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ..monitoring.time_it import timing
import dask.dataframe as dd
import dask
# filtering based on column values: function takes Dask- or Pandas-Dataframe

def calc_unique(series):
    return series.dropna().drop_duplicates()

@timing
def value_filter(df, only_null_or_empty=True, only_one=True, v_dask=True):

    cols2drop = []
    df = df.fillna(value=np.nan).replace('', np.nan).dropna(how='all')

    lazy_results = []
    if v_dask:
        for col in df.columns:
            series = df[col]
            lazy_result = dask.delayed(calc_unique)(series)
            lazy_results.append(lazy_result ) 
        lazy_results = dask.compute(*lazy_results)
    else:
        for col in df.columns:
            series = df[col]
            lazy_results.append(calc_unique(series))

    for col in lazy_results:
        if only_null_or_empty:
            if len(col) == 0:
                cols2drop.append(col.name)

        if only_one:
            if len(col) == 1:
                cols2drop.append(col.name)

    #if len(cols2drop) > 0:
        #df = df.drop(columns=cols2drop)

    return df