from ..monitoring.time_it import timing

@timing
def get_x_y(df, target_variables_column, errorcode):
    y = df[target_variables_column]
    y[y != errorcode] = 0
    y[y == errorcode] = 1
    y = y.astype(int)
    X = df.drop(columns=[target_variables_column])
    return (X, y)