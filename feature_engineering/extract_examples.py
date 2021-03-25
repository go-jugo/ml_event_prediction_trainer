import datetime
import pandas as pd
import dask.dataframe as dd
from sklearn.utils import shuffle
import dask
import glob
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from ..monitoring.time_it import timing
from math import ceil
import copy
import random
from .extract_windows_and_engineer_features_with_tsfresh import get_processed_timestamp_list
from .extract_windows_and_engineer_features_with_tsfresh import get_clean_errorcode_column_to_process
from .extract_windows_and_engineer_features_with_tsfresh import calculate_window


@timing
def extract_examples(df, error_code_series, errorcode_col, errorcode, pw_rw_list , minimal_features, iterations=1,
                     extract_examples=True):

    if extract_examples:

        for conf in pw_rw_list:
            print(conf)
            processed_timestamp_list_neg = get_processed_timestamp_list(errorcode, window_length=conf[1],
                                                                        window_end=conf[0], negative_examples=True)
            df_process_neg = get_clean_errorcode_column_to_process(error_code_series, errorcode_col, errorcode,
                                                                   window_end=conf[0], window_length=conf[1], negative_examples=True)
            df_process_neg = df_process_neg.drop(index=processed_timestamp_list_neg, errors='ignore')

            for i in range(iterations):
                df_process_neg = df_process_neg.drop(index=processed_timestamp_list_neg, errors='ignore')
                print('Number of possible examples to process: ' + str(len(df_process_neg)))
                if len(df_process_neg) >= 500:
                    df_loop = df_process_neg.sample(n=500)
                else:
                    df_loop = df_process_neg
                df_loop = df_loop.squeeze('columns')
                print('Number of examples to process this iteration: ' + str(len(df_loop)))
                process_list = list(zip(df_loop.index, df_loop))
                lazy_results = []
                for element in process_list:
                    window_start_date = element[0] - datetime.timedelta(seconds=(conf[1] + conf[0]))
                    window_end_date = element[0] - datetime.timedelta(seconds=(conf[0]))
                    lazy_result = dask.delayed(calculate_window)(df, window_start_date, window_end_date, element,
                                                                 minimal_features, window_length=conf[1],
                                                                 errorcode_col=errorcode_col, extract_negative_examples=True)
                    lazy_results.append(lazy_result)
                lazy_results = dask.compute(*lazy_results)
                df_tsfresh = pd.concat(lazy_results)
                processed_timestamp_list_neg.extend(df_tsfresh['global_timestamp'].to_list())
                df_tsfresh = df_tsfresh.dropna(axis=0, how='any')
                df_tsfresh = df_tsfresh.reset_index(drop=True)
                file_counter = len(glob.glob('../data/Extracted_Examples_ts_fresh/errorcode_' + str(errorcode) +
                                             '_PW_' + str(conf[0]) + '_RW_' + str(conf[1]) + '_' + 'neg*.gzip'))
                df_tsfresh.to_parquet('../data/Extracted_Examples_ts_fresh/errorcode_' + str(errorcode) + '_PW_' +
                                      str(conf[0]) + '_RW_' + str(conf[1]) + '_' + 'neg' + '_' +
                                      str(file_counter) + str('.parquet.gzip'))
                print('parquet created')


            processed_timestamp_list_pos = get_processed_timestamp_list(errorcode, window_length=conf[1], window_end=conf[0], negative_examples=False)
            df_process_pos = get_clean_errorcode_column_to_process(error_code_series, errorcode_col, errorcode,
                                                                   window_end=conf[0], window_length=conf[1],
                                                                   negative_examples=False)

            df_process_pos = df_process_pos.drop(index=processed_timestamp_list_pos, errors='ignore')
            print('Number of possible examples to process: ' + str(len(df_process_pos)))
            if len(df_process_pos) >= 500:
                df_loop = df_process_pos.sample(n=500)
            else:
                df_loop = df_process_pos
            df_loop = df_loop.squeeze('columns')
            print('Number of examples to process this iteration: ' + str(len(df_loop)))
            process_list = list(zip(df_loop.index, df_loop))
            lazy_results = []
            for element in process_list:
                window_start_date = element[0] - datetime.timedelta(seconds=(conf[1] + conf[0]))
                window_end_date = element[0] - datetime.timedelta(seconds=(conf[0]))
                lazy_result = dask.delayed(calculate_window)(df, window_start_date, window_end_date, element,
                                                             minimal_features, window_length=conf[1],
                                                             errorcode_col=errorcode_col, extract_negative_examples=False)
                lazy_results.append(lazy_result)
            lazy_results = dask.compute(*lazy_results)
            df_tsfresh = pd.concat(lazy_results)
            processed_timestamp_list_pos.extend(df_tsfresh['global_timestamp'].to_list())
            df_tsfresh = df_tsfresh.dropna(axis=0, how='any')
            df_tsfresh = df_tsfresh.reset_index(drop=True)
            file_counter = len(glob.glob('../data/Extracted_Examples_ts_fresh/errorcode_' + str(errorcode) +
                                         '_PW_' + str(conf[0]) + '_RW_' + str(conf[1]) + '_' + 'pos*.gzip'))
            df_tsfresh.to_parquet('../data/Extracted_Examples_ts_fresh/errorcode_' + str(errorcode) +
                                  '_PW_' + str(conf[0]) + '_RW_' + str(conf[1]) + '_' + 'pos' + '_' +
                                  str(file_counter) + str('.parquet.gzip'))
            print('parquet created')

    return df


