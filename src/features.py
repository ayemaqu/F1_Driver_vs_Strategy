# build the 5 engineered features from the cleaned core files
import pandas as pd
import os


# Era definitions, the single source of truth for the whole project.
# Notebooks 02, 04, and 05 import add_era from here instead of each
# re-typing the pd.cut bin edges.
    # All-caps at the top of a file signals "this is a constant, 
    # a fixed setting that never changes while the program runs."
ERA_BINS = [2009, 2013, 2020, 2021, 2024]
ERA_LABELS = ['2010-2013', '2014-2020', '2021', '2022-2024']


def add_era(df, year_col='year'):
    """
    Stamp every row with its era label based on the year column.

    This helper is the one place in the project where the era bin
    edges live. Any notebook or script that needs era imports this
    function so the definitions can never drift apart.
    """
    df = df.copy()
    df['era'] = pd.cut(
        df[year_col],
        bins=ERA_BINS,
        labels=ERA_LABELS
    )

    # Reproducibility check: every row landed in a bin, no year fell
    # outside the edges and came back null
    assert df['era'].isna().sum() == 0, \
        "add_era produced null eras, a year fell outside the bin edges"

    return df


def load_clean(clean_dir='../data/clean'):
    """
    Load the 5 cleaned core CSVs into a dictionary of DataFrames and
    recast the nullable Int64 columns lost in the CSV round-trip.

    CSV does not preserve nullable Int64 metadata, so position and
    number come back as float64 and must be recast on load. This
    mirrors the setup cells at the top of notebook 02.
    """
    file_names = ['results', 'races', 'pit_stops', 'drivers', 'constructors']

    dfs = {}
    for name in file_names:
        dfs[name] = pd.read_csv(f'{clean_dir}/{name}.csv')

    # Recast the two columns whose Int64 dtype the CSV round-trip drops.
    # duration_seconds needs no recast, it was born float64 and survives.
    dfs['results']['position'] = dfs['results']['position'].astype('Int64')
    dfs['drivers']['number'] = dfs['drivers']['number'].astype('Int64')

    # Reproducibility check: every table loaded and actually has rows.
    # read_csv crashes loudly on a missing file, but an empty file loads
    # quietly and would let the whole pipeline run hollow.
    for name, df in dfs.items():
        assert len(df) > 0, f"{name} loaded with zero rows"

    # Reproducibility check: the recasts took
    assert dfs['results']['position'].dtype == 'Int64', \
        "position is not nullable Int64 after reload"
    assert dfs['drivers']['number'].dtype == 'Int64', \
        "number is not nullable Int64 after reload"

    return dfs


def add_position_change(results):
    """
    Feature 1: position_change = grid - position. Driver skill proxy.

    DNF rows propagate null automatically (any math with pd.NA returns
    pd.NA). Pit lane starts (grid = 0) are set to null deliberately,
    because grid 0 is a code for pit lane, not a real grid slot, and
    the subtraction would produce a nonsense positions-gained number.
    """
    # create position_change
    results['position_change'] = results['grid'] - results['position']

    # Pit lane starts get null, grid = 0 is a code, not a position
    results.loc[results['grid'] == 0, 'position_change'] = pd.NA

    # Reproducibility check: every pit lane start ended up null
    pit_lane = results[results['grid'] == 0]
    assert pit_lane['position_change'].isna().all(), \
        "A pit lane start (grid = 0) kept a position_change value"

    # Reproducibility check: every DNF ended up null (null propagation held)
    dnf = results[results['position'].isna()]
    assert dnf['position_change'].isna().all(), \
        "A DNF row has a position_change value, null propagation broke"

    return results


def add_year_and_era(results, races):
    """
    Feature 2: era. Merge year from races, then bin into the four eras.

    year lives in races, not results. The merge pulls only raceId and
    year so results stays lean. Era is stamped by the add_era helper,
    the single source of truth for the bin edges.
    """
    rows_before = len(results)

    # Merge the year from the races into the results dataframe
    results = results.merge(races[['raceId', 'year']], on='raceId')

    # Reproducibility check: one-to-one join, no duplication or drops
    assert len(results) == rows_before, \
        "Row count changed after the year merge, the join is not one-to-one"

    # Era stamping and its null assert both live inside add_era
    results = add_era(results)

    return results


def build_driver_season(results):
    """
    Feature 3: points_per_race. New driver_season table, one row per
    driver per season. Driver skill proxy, fairness-adjusted.

    Rows where the driver never took the start (W withdrew, F failed
    to qualify, E excluded) leave the denominator before aggregation.
    R, D, and N rows stay, because those drivers raced and scoring
    zero from a start is honest.
    """
    # Exclude rows where the driver never started: W, F, E
    results_started = results[~results['positionText'].isin(['W', 'F', 'E'])].copy()

    # Group by driver and season, total points and races started per pile
    driver_season = (
        results_started
        .groupby(['driverId', 'year'])
        .agg(
            total_points=('points', 'sum'),
            races_started=('raceId', 'nunique')
        )
        .reset_index()
    )

    # The feature itself
    driver_season['points_per_race'] = driver_season['total_points'] / driver_season['races_started']

    # Reproducibility check: no W, F, or E rows made it into the source
    assert not results_started['positionText'].isin(['W', 'F', 'E']).any(), \
        "A never-started row (W/F/E) survived the exclusion filter"

    # Reproducibility check: every driver-season started at least one
    # race, so the division can never blow up or produce infinity
    assert (driver_season['races_started'] >= 1).all(), \
        "A driver-season has zero races started"

    # Reproducibility check: the new table is fully populated
    assert driver_season.isna().sum().sum() == 0, \
        "driver_season has nulls"

    return driver_season


def build_team_season(results):
    """
    Feature 4: teammate_gap. New team_season table, one row per team
    per season. Driver skill proxy, the strongest one: same car,
    different results points to the driver.

    Three-step build: pile table per team-driver-season, top-two
    eviction (keep the two drivers who entered the most races, season
    points as tiebreaker), then gap = higher scorer minus lower scorer.
    driverId is dropped on purpose, max/min aggregation loses identity.
    """
    # Same started-only base as Feature 3, consistent denominators
    results_started = results[~results['positionText'].isin(['W', 'F', 'E'])].copy()

    # Pile table: one row per team-driver-season
    team_driver_season = (
        results_started
        .groupby(['constructorId', 'driverId', 'year'])
        .agg(
            season_points=('points', 'sum'),
            races_entered=('raceId', 'nunique')
        )
        .reset_index()
    )

    # Top-two eviction: teams field 2 cars, but mid-season swaps mean
    # some team-seasons show 3+ drivers. Keep the two who entered the
    # most races, season_points breaks ties for determinism.
    top_two = (
        team_driver_season
        .sort_values(['races_entered', 'season_points'], ascending=False)
        .groupby(['constructorId', 'year'])
        .head(2)
    )

    # Reproducibility check: the eviction actually held, no team-season
    # carries more than two drivers into the gap calculation
    assert top_two.groupby(['constructorId', 'year']).size().max() <= 2, \
        "A team-season kept more than two drivers after eviction"

    # For each team-season, the gap is the higher scorer minus the lower
    team_season = (
        top_two
        .groupby(['constructorId', 'year'])
        .agg(
            points_high=('season_points', 'max'),
            points_low=('season_points', 'min')
        )
        .reset_index()
    )

    # The feature itself
    team_season['teammate_gap'] = team_season['points_high'] - team_season['points_low']

    # Reproducibility check: max minus min can never be negative. If it
    # ever is, the aggregation logic broke.
    assert (team_season['teammate_gap'] >= 0).all(), \
        "A negative teammate_gap appeared, points_high < points_low"

    return team_season


def build_avg_pit_stop(pit_stops, results):
    """
    Feature 5: avg_pit_stop_duration. New avg_pit_stop table, one row
    per constructor per season. Team strategy proxy, primary.

    pit_stops has no constructorId or year, so both are borrowed from
    results. The merge needs BOTH keys: raceId alone matches ~20
    drivers per race, driverId alone matches one driver across
    hundreds of races. Together they pinpoint exactly one row.

    Stops of 60+ seconds are excluded before averaging. Those are race
    stoppage parking events (red flag or safety), not crew service.
    Documented data audit correction, including them shifts team-season
    averages by up to 225 seconds.
    """
    rows_before = len(pit_stops)

    # Borrow constructorId and year so pit stops know their team
    pit_stops = pit_stops.merge(
        results[['raceId', 'driverId', 'constructorId', 'year']],
        on=['raceId', 'driverId'], how='left'
    )

    # Reproducibility check: the two-key join matched one row each,
    # no explosion, no drops
    assert len(pit_stops) == rows_before, \
        "Row count changed after the pit stop merge, join keys are wrong"

    # Reproducibility check: every pit stop found its team
    assert pit_stops['constructorId'].isna().sum() == 0, \
        "A pit stop failed to match a race entry, orphan rows exist"

    # Deliberate exclusion: 60+ second stops are parking, not service
    pit_stops_service = pit_stops[pit_stops['duration_seconds'] < 60].copy()

    # Pile step: average crew service duration per team per season
    avg_pit_stop = (
        pit_stops_service
        .groupby(['constructorId', 'year'])
        .agg(avg_pit_stop_duration=('duration_seconds', 'mean'))
        .reset_index()
    )

    # Reproducibility check: the exclusion rule held, no average was
    # built from a parking event
    assert (avg_pit_stop['avg_pit_stop_duration'] < 60).all(), \
        "An average pit stop is 60+ seconds, the exclusion rule failed"

    # Reproducibility check: no 2010 rows. Pit stop data starts 2011,
    # a 2010 row here means the structural gap got filled with garbage.
    assert (avg_pit_stop['year'] == 2010).sum() == 0, \
        "2010 rows appeared in avg_pit_stop, pit data should start 2011"

    # Reproducibility check: fully populated
    assert avg_pit_stop['avg_pit_stop_duration'].isna().sum() == 0, \
        "avg_pit_stop_duration has nulls"

    return avg_pit_stop


def export_processed(tables, out_dir='../data/processed'):
    """
    Write the feature tables to CSV in the processed/ folder.

    results gets re-exported because position_change, year, and era
    were added to it. driver_season, team_season, and avg_pit_stop are
    new tables born in this script.
    """
    for name, df in tables.items():
        df.to_csv(f'{out_dir}/{name}.csv', index=False)

    # Reproducibility check: every expected file now exists on disk.
    # Honest limit, same as cleaning.py: checks existence, not content.
    for name in tables:
        expected_path = f'{out_dir}/{name}.csv'
        assert os.path.exists(expected_path), \
            f"Expected output file was not written: {expected_path}"

    return tables


def main():
    """Run the full feature engineering pipeline in order."""
    dfs = load_clean()

    results = dfs['results']
    races = dfs['races']
    pit_stops = dfs['pit_stops']

    # Order matters: position_change needs nothing extra, era needs the
    # year merge, and every table after that needs year to exist.
    results = add_position_change(results)
    results = add_year_and_era(results, races)
    driver_season = build_driver_season(results)
    team_season = build_team_season(results)
    avg_pit_stop = build_avg_pit_stop(pit_stops, results)

    tables = {
        'results': results,
        'driver_season': driver_season,
        'team_season': team_season,
        'avg_pit_stop': avg_pit_stop
    }
    export_processed(tables)
    print("Feature pipeline finished. Processed files written to data/processed.")
    return tables


if __name__ == '__main__':
    main()