from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def get_config_path(file_name):
    return CONFIG_DIR / file_name


def get_raw_league_dir(league_code):
    return RAW_DATA_DIR / league_code


def get_raw_league_seasons_dir(league_code):
    return RAW_DATA_DIR / league_code / "seasons"


def get_processed_league_dir(league_code):
    return PROCESSED_DATA_DIR / league_code


def get_outputs_league_dir(league_code):
    return OUTPUTS_DIR / league_code


def get_market_output_dir(league_code, market_name):
    return OUTPUTS_DIR / league_code / market_name


def get_market_models_dir(league_code, market_name):
    return get_market_output_dir(league_code, market_name) / "models"


def get_market_predictions_dir(league_code, market_name):
    return get_market_output_dir(league_code, market_name) / "predictions"


def get_market_value_scans_dir(league_code, market_name):
    return get_market_output_dir(league_code, market_name) / "value_scans"


def get_market_paper_tracking_dir(league_code, market_name):
    return get_market_output_dir(league_code, market_name) / "paper_tracking"


def get_market_dashboard_dir(league_code, market_name):
    return get_market_output_dir(league_code, market_name) / "dashboard"


def get_market_dashboard_games_dir(league_code, market_name):
    return get_market_dashboard_dir(league_code, market_name) / "games"


def get_global_dashboard_dir():
    return OUTPUTS_DIR / "dashboard"


def get_league_matches_path(league_code):
    return get_processed_league_dir(league_code) / f"{league_code}_matches.csv"


def get_market_dataset_path(league_code, market_name):
    return get_processed_league_dir(league_code) / f"{league_code}_{market_name}.csv"


def get_market_features_path(league_code, market_name):
    return get_processed_league_dir(league_code) / f"{league_code}_{market_name}_features.csv"


def get_nested_predictions_path(league_code, market_name):
    return get_market_predictions_dir(league_code, market_name) / "nested_feature_selection_predictions.csv"


def get_nested_by_year_path(league_code, market_name):
    return get_market_predictions_dir(league_code, market_name) / "nested_feature_selection_by_year.csv"


def get_value_scan_candidates_path(league_code, market_name):
    return get_market_value_scans_dir(league_code, market_name) / "value_scan_candidates.csv"


def get_value_scan_bets_path(league_code, market_name):
    return get_market_value_scans_dir(league_code, market_name) / "value_scan_bets.csv"


def get_value_scan_by_year_path(league_code, market_name):
    return get_market_value_scans_dir(league_code, market_name) / "value_scan_by_year.csv"


def ensure_base_dirs_for_league_market(league_code, market_name):
    directories = [
        get_raw_league_dir(league_code),
        get_raw_league_seasons_dir(league_code),
        get_processed_league_dir(league_code),
        get_market_output_dir(league_code, market_name),
        get_market_models_dir(league_code, market_name),
        get_market_predictions_dir(league_code, market_name),
        get_market_value_scans_dir(league_code, market_name),
        get_market_paper_tracking_dir(league_code, market_name),
        get_market_dashboard_dir(league_code, market_name),
        get_market_dashboard_games_dir(league_code, market_name),
        get_global_dashboard_dir(),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
