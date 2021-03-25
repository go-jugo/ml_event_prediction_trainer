import dask.dataframe as dd

def descibe_dataframe(df):
    error_cols = [col for col in df.columns if 'errorCode' in col]
    df = df.categorize(columns=error_cols)
    statistics = df.describe(include='all').compute()
    statistics = statistics.T
    statistics.to_excel('../statistics/variables_statistics.xlsx')