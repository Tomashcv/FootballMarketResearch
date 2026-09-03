import argparse
import json
from pathlib import Path

import requests

from src.common.paths import get_config_path
from src.common.paths import get_raw_league_seasons_dir


def load_leagues_config():
    config_path = get_config_path("leagues.json")

    with open(config_path, "r", encoding="utf-8") as file:
        leagues = json.load(file)

    return leagues


def season_to_football_data_code(season_start_year):
    season_end_year = season_start_year + 1

    start_short = str(season_start_year)[-2:]
    end_short = str(season_end_year)[-2:]

    return start_short + end_short


def build_download_url(season_start_year, football_data_code):
    season_code = season_to_football_data_code(season_start_year)
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{football_data_code}.csv"


def existing_csv_files(directory):
    if not directory.exists():
        return []

    files = sorted(directory.glob("*.csv"))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    league_code = args.league.upper()

    leagues = load_leagues_config()

    if league_code not in leagues:
        raise ValueError(f"Liga não encontrada em config/leagues.json: {league_code}")

    league_config = leagues[league_code]
    football_data_code = league_config["football_data_code"]

    output_dir = get_raw_league_seasons_dir(league_code)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_files = existing_csv_files(output_dir)

    if len(current_files) > 0 and not args.force:
        print(f"Já existem {len(current_files)} CSVs em {output_dir}")
        print("Não vou fazer download porque --force não foi usado.")
        print("Ficheiros encontrados:")

        for file_path in current_files:
            print("-", file_path.name)

        return

    for season_start_year in range(args.start_year, args.end_year + 1):
        season_end_year = season_start_year + 1

        output_name = f"{league_code}_{season_start_year}_{season_end_year}.csv"
        output_path = output_dir / output_name

        if output_path.exists() and not args.force:
            print(f"Skip, já existe: {output_path}")
            continue

        url = build_download_url(season_start_year, football_data_code)

        print(f"A descarregar {league_code} {season_start_year}/{season_end_year}: {url}")

        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            print(f"Falhou {season_start_year}/{season_end_year}: HTTP {response.status_code}")
            continue

        text = response.text.strip()

        if len(text) == 0:
            print(f"Falhou {season_start_year}/{season_end_year}: ficheiro vazio")
            continue

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(response.text)

        print(f"Guardado: {output_path}")


if __name__ == "__main__":
    main()
