from ..tools.series_list_to_df import series_list_to_df

def create_error_code_col(df, error_code_col):
    df = df[[error_code_col]]
    return df