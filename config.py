from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json
import os


default_scaler = os.getenv("ML_DEFAULT_SCALER", "StandardScaler")
default_ml_algorithm = os.getenv("ML_DEFAULT_ALGORITHM", "RandomForestClassifier")
logging_level = os.getenv("ML_LOGGING_LEVEL", "info")
logging_color = int(os.getenv("ML_LOGGING_COLOR", 0))
v_dask = int(os.getenv("ML_V_DASK", 1))

debug_mode = True if logging_level == "debug" else False
write_monitoring = False
store_results = False


ml_algorithm_map = {
    "RandomForestClassifier": RandomForestClassifier
}

scaler_map = {
    "StandardScaler": StandardScaler
}


base_config = json.loads(os.getenv("ML_BASE_CONFIG")) if os.getenv("ML_BASE_CONFIG") else dict(
    sampling_frequency=["5S"],
    imputations_technique_str=["pad"],
    imputation_technique_num=["pad"],
    ts_fresh_window_length=[30],
    ts_fresh_window_end=[30],
    ts_fresh_minimal_features=[True],
    balance_ratio=[0.5],
    random_state=[[0]],
    cv=[5],
    oversampling_method=[False],
)


def configs_from_dict(config: dict) -> list:
    conf = base_config.copy()
    conf.update(config)
    if "scaler" in conf:
        conf["scaler"] = [scaler_map[scaler]() for scaler in conf["scaler"].copy()]
    else:
        conf["scaler"] = [scaler_map[default_scaler]()]
    if "ml_algorithm" in conf:
        conf["ml_algorithm"] = [ml_algorithm_map[ml_algorithm]() for ml_algorithm in conf["ml_algorithm"].copy()]
    else:
        conf["ml_algorithm"] = [ml_algorithm_map[default_ml_algorithm]()]
    return [dict(zip(conf, v)) for v in product(*conf.values())]


def configs_from_json(config: str) -> list:
    return configs_from_dict(json.loads(config))


def config_from_dict(config: dict) -> dict:
    config = config.copy()
    config["scaler"] = scaler_map[config["scaler"]]()
    config["ml_algorithm"] = ml_algorithm_map[config["ml_algorithm"]]()
    return config


def config_from_json(config: str) -> dict:
    return config_from_dict(json.loads(config))


def create_configs():
    base_config = dict(
        sampling_frequency=['5S'],
        imputations_technique_str=['pad'],
        imputation_technique_num=['pad'],
        ts_fresh_window_length=[30],
        ts_fresh_window_end=[30],
        ts_fresh_minimal_features=[True],
        scaler=[StandardScaler()],
        target_col=['module_2_errorcode'],
        target_errorCode=[1051],
        balance_ratio = [0.5],
        random_state = [[0]],
        cv=[5],
        oversampling_method = [False],
        ml_algorithm=[RandomForestClassifier()]
        )
    configs_pipeline = [dict(zip(base_config, v)) for v in product(*base_config.values())]
    return configs_pipeline

configs_pipeline = create_configs()

