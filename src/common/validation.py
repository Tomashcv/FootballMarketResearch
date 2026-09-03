
def get_sorted_years(dataframe, year_column="season_end_year"):
    years = sorted(dataframe[year_column].dropna().unique().tolist())
    return years


def get_walk_forward_test_years(dataframe, year_column="season_end_year", min_train_years=5):
    years = get_sorted_years(dataframe, year_column)

    if len(years) <= min_train_years:
        return []

    return years[min_train_years:]


def split_train_test_by_year(dataframe, test_year, year_column="season_end_year"):
    train_df = dataframe[dataframe[year_column] < test_year].copy()
    test_df = dataframe[dataframe[year_column] == test_year].copy()

    return train_df, test_df
