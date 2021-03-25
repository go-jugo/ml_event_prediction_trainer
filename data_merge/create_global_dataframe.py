import glob
import pandas as pd
import dask.dataframe as dd
from ..monitoring.time_it import timing
from ..tools.dask_repartition import dask_repartition
import random

@timing
def create_global_dataframe( buffer_data_path, v_dask=True, draw_sample=False, sample_size=0, dataset = 'CPT'):
    files = glob.glob(buffer_data_path)
    if dataset == 'SITEC':
        df = create_sitec_df(files)
        print(df.columns)
        return df
    if draw_sample:
        files = random.sample(files, sample_size)
    if v_dask:
        print(files)
        cols2read = compute_col_union(files)
        df = dd.read_parquet(files, columns=cols2read)
        df = dask_repartition(df)
        print('Setting index ...')
        df = df.set_index('global_timestamp')
        df = dask_repartition(df)
        df['global_timestamp'] = df.index
        print('Removing duplicates ...')
        df = df.map_partitions(lambda d: d.drop_duplicates(subset='global_timestamp'))
        df = df.drop('global_timestamp', axis=1)
        #df = df.drop_duplicates(subset=['global_timestamp'], split_every=10000)
        df = dask_repartition(df)
        object_column_list = list(df.select_dtypes(include='object').columns)
        df[object_column_list] = df[object_column_list].astype(str)
        #df = df[~df.index.to_series().duplicated()]
    else:
        df = pd.concat([pd.read_parquet(fp) for fp in file_path_list], ignore_index=True)
    return df

def create_sitec_df(files):
    for i, file in enumerate(files):
        if i == 0:
            df = dd.read_parquet(file)  
        elif i >= 1:
            tmp_df = dd.read_parquet(file)
            df = dd.concat([df, tmp_df])
    df = dask_repartition(df)
    df = df.set_index('global_timestamp')
    df = dask_repartition(df)
    return df

def compute_col_union(file_path_list):
    cols_global = []
    for file in file_path_list:
        cols_global.append(list(dd.read_parquet(file).columns))
    cols_union = sorted(list(set(cols_global[0]).intersection(*cols_global)))
    return cols_union