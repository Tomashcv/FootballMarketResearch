from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NestedYearFold:
    test_year: int
    tune_year: int
    calibration_year: int
    train_year_max: int


def build_nested_year_folds(
    frame: pd.DataFrame,
    test_years: list[int],
    tune_lag_years: int = 2,
    calibration_lag_years: int = 1,
    min_train_rows: int = 1000,
) -> list[NestedYearFold]:
    if "season_start_year" not in frame.columns:
        raise KeyError("season_start_year is required")

    years = pd.to_numeric(frame["season_start_year"], errors="coerce")
    folds: list[NestedYearFold] = []

    for test_year in sorted(set(int(y) for y in test_years)):
        tune_year = test_year - int(tune_lag_years)
        calibration_year = test_year - int(calibration_lag_years)

        train_mask = years.lt(tune_year)
        tune_mask = years.eq(tune_year)
        calibration_mask = years.eq(calibration_year)
        test_mask = years.eq(test_year)

        if int(train_mask.sum()) < min_train_rows:
            continue
        if not tune_mask.any() or not calibration_mask.any() or not test_mask.any():
            continue

        folds.append(
            NestedYearFold(
                test_year=test_year,
                tune_year=tune_year,
                calibration_year=calibration_year,
                train_year_max=tune_year - 1,
            )
        )

    return folds


def _validated_dates(part: pd.DataFrame, partition_name: str) -> pd.Series:
    if part.empty:
        raise ValueError(f"Temporal partition {partition_name} is empty")
    if "match_date" not in part.columns:
        raise KeyError("match_date is required for leakage-safe temporal splitting")

    dates = pd.to_datetime(part["match_date"], errors="coerce")

    invalid_count = int(dates.isna().sum())
    if invalid_count:
        raise ValueError(
            f"Temporal partition {partition_name} contains "
            f"{invalid_count} invalid match_date values"
        )

    return dates


def _purge_before(
    part: pd.DataFrame,
    cutoff: pd.Timestamp,
    partition_name: str,
) -> pd.DataFrame:
    """
    Keep only rows strictly earlier than the next temporal partition.

    This handles cross-league calendar overlaps, delayed fixtures and seasons
    such as 2019/20, without allowing later outcomes into an earlier stage.
    Same-date rows are also removed because most source rows only provide a
    date and cannot prove an intraday ordering.
    """
    dates = _validated_dates(part, partition_name)
    keep = dates.lt(cutoff)

    purged = part.loc[keep].copy()
    if purged.empty:
        raise ValueError(
            f"Temporal partition {partition_name} became empty after "
            f"purging rows on or after {cutoff}"
        )

    purged["_temporal_sort_date"] = pd.to_datetime(
        purged["match_date"], errors="raise"
    )
    purged = (
        purged.sort_values("_temporal_sort_date", kind="stable")
        .drop(columns="_temporal_sort_date")
        .reset_index(drop=True)
    )
    return purged


def split_nested_fold(
    frame: pd.DataFrame,
    fold: NestedYearFold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = pd.to_numeric(frame["season_start_year"], errors="coerce")

    train = frame[years.lt(fold.tune_year)].copy()
    tune = frame[years.eq(fold.tune_year)].copy()
    calibration = frame[years.eq(fold.calibration_year)].copy()
    test = frame[years.eq(fold.test_year)].copy()

    # Purge backwards from the untouched test partition. This preserves the
    # declared season folds while guaranteeing strict global chronology.
    test_start = _validated_dates(test, "test").min()

    calibration = _purge_before(
        calibration,
        test_start,
        "calibration",
    )
    calibration_start = _validated_dates(
        calibration,
        "calibration",
    ).min()

    tune = _purge_before(
        tune,
        calibration_start,
        "tune",
    )
    tune_start = _validated_dates(tune, "tune").min()

    train = _purge_before(
        train,
        tune_start,
        "train",
    )

    assert_temporal_order(train, tune, calibration, test)
    return train, tune, calibration, test


def assert_temporal_order(
    train: pd.DataFrame,
    tune: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    partitions = {
        "train": train,
        "tune": tune,
        "calibration": calibration,
        "test": test,
    }

    years: dict[str, tuple[int, int]] = {}
    for name, part in partitions.items():
        values = (
            pd.to_numeric(part["season_start_year"], errors="coerce")
            .dropna()
            .astype(int)
        )
        if values.empty:
            raise ValueError(f"Temporal partition {name} is empty")

        years[name] = (int(values.min()), int(values.max()))

    if not (
        years["train"][1]
        < years["tune"][0]
        < years["calibration"][0]
        < years["test"][0]
    ):
        raise AssertionError(f"Temporal season order violated: {years}")

    date_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, part in partitions.items():
        dates = _validated_dates(part, name)
        date_ranges[name] = (dates.min(), dates.max())

    if not (
        date_ranges["train"][1] < date_ranges["tune"][0]
        and date_ranges["tune"][1] < date_ranges["calibration"][0]
        and date_ranges["calibration"][1] < date_ranges["test"][0]
    ):
        raise AssertionError(f"Date order violated after purge: {date_ranges}")
