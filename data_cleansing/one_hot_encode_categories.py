import pandas as pd
import dask.dataframe as dd
from ..monitoring.time_it import timing
from ..tools.dask_repartition import dask_repartition
from ..logger import get_logger

logger = get_logger(__name__.split(".", 1)[-1])


@timing
def one_hot_encode_categories(df, errorcode_col, v_dask=True):

    non_numeric_columns_list = list(df.select_dtypes(exclude=['number', 'datetime']).columns)

    df_non_numeric = df[non_numeric_columns_list].astype(str)
    if v_dask:
        df_non_numeric = df_non_numeric.categorize()
    if len(df_non_numeric.columns) != 0:
        df_dummy = dd.get_dummies(df_non_numeric, prefix_sep='.', prefix=non_numeric_columns_list)
        df = df.drop(columns=non_numeric_columns_list)
        if v_dask:
            df = dd.concat([df, df_dummy], axis=1)
        else:
            df = pd.concat([df, df_dummy], axis=1)
        logger.debug('Number of Columns for one hot encoding : ' + str(len(non_numeric_columns_list)))

    df = dask_repartition(df)

    return df