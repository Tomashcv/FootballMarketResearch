import argparse
import re

import pandas as pd
from pandas.errors import ParserError

from src.common.paths import get_raw_league_seasons_dir
from src.common.paths import get_league_matches_path


ENCODINGS_TO_TRY = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]


def extract_season_from_filename(file_path):
    file_name = file_path.stem

    match = re.search(r"(\d{4})[_-](\d{4})", file_name)

    if match is not None:
        season_start_year = int(match.group(1))
        season_end_year = int(match.group(2))
        return season_start_year, season_end_year

    return None, None


def clean_column_names(dataframe):
    cleaned_columns = []

    for column in dataframe.columns:
        cleaned_column = str(column)
        cleaned_column = cleaned_column.strip()
        cleaned_column = cleaned_column.replace("\ufeff", "")
        cleaned_column = cleaned_column.replace("ï»¿", "")
        cleaned_column = cleaned_column.replace("\xa0", " ")
        cleaned_column = cleaned_column.strip()
        cleaned_columns.append(cleaned_column)

    dataframe.columns = cleaned_columns

    return dataframe


def read_raw_csv_strict(file_path):
    last_error = None

    for encoding in ENCODINGS_TO_TRY:
        try:
            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False
            )

            dataframe = clean_column_names(dataframe)

            return dataframe, encoding, "strict", 0

        except UnicodeDecodeError as error:
            last_error = error

        except ParserError as error:
            last_error = error

    raise last_error


def count_raw_lines(file_path, encoding):
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as file:
            return sum(1 for _ in file)
    except Exception:
        return None


def read_raw_csv_flexible(file_path):
    last_error = None

    for encoding in ENCODINGS_TO_TRY:
        try:
            raw_lines = count_raw_lines(file_path, encoding)

            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
                engine="python",
                on_bad_lines="skip"
            )

            dataframe = clean_column_names(dataframe)

            skipped_lines = 0

            if raw_lines is not None:
                # -1 por causa do header
                expected_data_rows = max(raw_lines - 1, 0)
                skipped_lines = max(expected_data_rows - len(dataframe), 0)

            return dataframe, encoding, "flexible_skip_bad_lines", skipped_lines

        except UnicodeDecodeError as error:
            last_error = error

        except ParserError as error:
            last_error = error

    raise last_error


def read_raw_csv(file_path):
    try:
        return read_raw_csv_strict(file_path)
    except Exception:
        return read_raw_csv_flexible(file_path)


def parse_match_date(dataframe):
    if "Date" not in dataframe.columns:
        raise ValueError("Coluna Date não encontrada.")

    try:
        dataframe["Date"] = pd.to_datetime(
            dataframe["Date"],
            dayfirst=True,
            errors="coerce",
            format="mixed"
        )
    except TypeError:
        dataframe["Date"] = pd.to_datetime(
            dataframe["Date"],
            dayfirst=True,
            errors="coerce"
        )

    return dataframe


def clean_required_rows(dataframe):
    required_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    for column in required_columns:
        if column not in dataframe.columns:
            print("Colunas disponíveis:")
            print(list(dataframe.columns))
            raise ValueError(f"Coluna obrigatória em falta: {column}")

    before = len(dataframe)
    dataframe = dataframe.dropna(subset=required_columns).copy()
    after = len(dataframe)

    removed = before - after

    return dataframe, removed


def infer_season_from_dates(dataframe):
    if len(dataframe) == 0:
        return None, None

    min_date = dataframe["Date"].min()
    max_date = dataframe["Date"].max()

    if pd.isna(min_date) or pd.isna(max_date):
        return None, None

    season_start_year = int(min_date.year)
    season_end_year = int(max_date.year)

    return season_start_year, season_end_year


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    args = parser.parse_args()

    league_code = args.league.upper()

    raw_dir = get_raw_league_seasons_dir(league_code)
    output_path = get_league_matches_path(league_code)

    csv_files = sorted(raw_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"Não encontrei CSVs em {raw_dir}")

    all_dataframes = []

    total_bad_lines_skipped = 0
    total_required_rows_removed = 0

    for file_path in csv_files:
        print(f"A ler: {file_path}")

        season_start_year, season_end_year = extract_season_from_filename(file_path)

        dataframe, encoding, read_mode, skipped_lines = read_raw_csv(file_path)

        print(f"  encoding usado: {encoding}")
        print(f"  modo leitura: {read_mode}")

        if skipped_lines > 0:
            print(f"  linhas más ignoradas: {skipped_lines}")

        total_bad_lines_skipped += skipped_lines

        dataframe = parse_match_date(dataframe)
        dataframe, required_rows_removed = clean_required_rows(dataframe)

        total_required_rows_removed += required_rows_removed

        inferred_start_year, inferred_end_year = infer_season_from_dates(dataframe)

        if season_start_year is None:
            season_start_year = inferred_start_year

        if season_end_year is None:
            season_end_year = inferred_end_year

        dataframe["league"] = league_code
        dataframe["season_start_year"] = season_start_year
        dataframe["season_end_year"] = season_end_year
        dataframe["source_file"] = file_path.name

        print(f"  época inferida: {season_start_year}/{season_end_year}")
        print(f"  jogos válidos: {len(dataframe)}")

        all_dataframes.append(dataframe)

    final_dataframe = pd.concat(all_dataframes, ignore_index=True)
    final_dataframe = final_dataframe.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    before_duplicates = len(final_dataframe)

    final_dataframe = final_dataframe.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"],
        keep="first"
    ).copy()

    after_duplicates = len(final_dataframe)
    duplicates_removed = before_duplicates - after_duplicates

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataframe.to_csv(output_path, index=False)

    print("")
    print("Dataset criado:")
    print(output_path)
    print("Jogos:", len(final_dataframe))
    print("Duplicados removidos:", duplicates_removed)
    print("Linhas más ignoradas:", total_bad_lines_skipped)
    print("Linhas removidas por campos obrigatórios em falta:", total_required_rows_removed)
    print("Épocas:", sorted(final_dataframe["season_end_year"].dropna().astype(int).unique().tolist()))
    print("Colunas:", len(final_dataframe.columns))


if __name__ == "__main__":
    main()
