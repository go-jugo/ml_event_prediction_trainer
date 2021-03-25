import pandas as pd
import dask.dataframe as dd
from .dask_repartition import dask_repartition
from math import floor
from ..logger import get_logger

logger = get_logger(__name__.split(".", 1)[-1])


def series_list_to_df(series_collection, v_dask=True):
    df = pd.DataFrame({series_collection[0].name: series_collection[0]})
    if v_dask:
        npartitions_calc = floor(len(df)/100000)+1
        logger.debug('calculated partitions: ' + str(npartitions_calc))
        df = dd.from_pandas(df, npartitions = npartitions_calc)
    for series in series_collection[1:]:
        name = series.name
        df[name] = series   
    df = dask_repartition(df)
    return (df)