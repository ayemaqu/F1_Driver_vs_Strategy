# load all csvs from raw core files 
import pandas as pd
import os


def load_raw(raw_dir='../data/raw'):
    """
    Load the 5 core raw CSVs into a dictionary of DataFrames.

    Returns a dict keyed by table name so every downstream cleaning step can loop over the same structure
    """
    file_names = ['results', 'races', 'pit_stops', 'drivers', 'constructors']

    dfs = {}
    for name in file_names:
        dfs[name] = pd.read_csv(f'{raw_dir}/{name}.csv')

    return dfs


def filter_years(dfs, start=2010, end=2024):
    """
    Filter all 5 core tables to the analysis window.

    Only races has a year column, so the filter flows outward through
    raceId, then driverId and constructorId, mirroring notebook 01.
    """
    # Filter races directly on year
    dfs['races'] = dfs['races'][(dfs['races']['year'] >= start) & (dfs['races']['year'] <= end)].copy()

    # Valid race IDs come from the filtered races
    valid_race_ids = dfs['races']['raceId']

    # Filter the race-keyed tables
    dfs['results'] = dfs['results'][dfs['results']['raceId'].isin(valid_race_ids)].copy()
    dfs['pit_stops'] = dfs['pit_stops'][dfs['pit_stops']['raceId'].isin(valid_race_ids)].copy()

    # Filter the lookup tables off who actually appears in the filtered results
    dfs['drivers'] = dfs['drivers'][dfs['drivers']['driverId'].isin(dfs['results']['driverId'])].copy()
    dfs['constructors'] = dfs['constructors'][dfs['constructors']['constructorId'].isin(dfs['results']['constructorId'])].copy()

    # Reproducibility check: every surviving race is inside the window
    assert dfs['races']['year'].between(start, end).all(), \
        "A race outside the year window survived filtering"

    return dfs    

    
def replace_null_placeholders(dfs):
    """
    Replace the text '\\N' placeholders with real nulls (pd.NA).

    The raw Kaggle CSVs store missing values as the literal string '\\N',
    so pandas reads them as valid text and every column looks null-free
    until this runs.
    """
    for name, df in dfs.items():
        dfs[name] = df.replace('\\N', pd.NA)

    return dfs

# Column definitions, directly from notebook 01
KEEP_COLUMNS = {
    'results': ['raceId', 'driverId', 'constructorId', 'grid',
                'position', 'positionText', 'points', 'statusId'],
    'races': ['raceId', 'year', 'circuitId', 'name'],
    'pit_stops': ['raceId', 'driverId', 'stop', 'lap', 'duration', 'milliseconds'],
    'drivers': ['driverId', 'driverRef', 'number', 'code',
                'forename', 'surname', 'nationality'],
    'constructors': ['constructorId', 'constructorRef', 'name', 'nationality'],
}

RENAME_MAP = {
    'races': {'name': 'race_name'},
    'constructors': {'name': 'constructor_name'},
}


def trim_columns(dfs, keep_columns=KEEP_COLUMNS):
    """Reduce each table to only the columns needed for analysis."""
    for name, cols in keep_columns.items():
        dfs[name] = dfs[name][cols].copy()

    # Reproducibility check: each table has exactly the columns we asked for
    for name, cols in keep_columns.items():
        assert list(dfs[name].columns) == cols, \
            f"{name} columns do not match keep_columns after trim"

    return dfs


def rename_columns(dfs, rename_map=RENAME_MAP):
    """Rename ambiguous 'name' columns so joins do not collide."""
    for name, mapping in rename_map.items():
        dfs[name] = dfs[name].rename(columns=mapping)

    # Reproducibility check: the new name is present, the old one is gone
    for name, mapping in rename_map.items():
        for old, new in mapping.items():
            assert new in dfs[name].columns, \
                f"{name} is missing renamed column '{new}'"
            assert old not in dfs[name].columns, \
                f"{name} still has old column '{old}' after rename"

    return dfs


def fix_dtypes(dfs):
    """
    Convert string columns to their correct numeric types.

    position and number become nullable Int64 (whole numbers that must
    keep their structural nulls). duration_seconds is built from
    milliseconds, because duration strings over 60 seconds are stored
    as MM:SS.mmm and cannot be parsed by pd.to_numeric.
    """
    # Coerce the string columns to numbers first. errors='coerce' turns
    # any unparseable value into a null instead of raising.
    dfs['results']['position'] = pd.to_numeric(
        dfs['results']['position'], errors='coerce'
    )
    dfs['drivers']['number'] = pd.to_numeric(
        dfs['drivers']['number'], errors='coerce'
    )

    # Build duration_seconds from milliseconds, which holds every value
    # including the 60+ second red-flag stops. The duration string column
    # stays as human-readable evidence.
    dfs['pit_stops']['duration_seconds'] = dfs['pit_stops']['milliseconds'] / 1000

    # Lock whole-number columns to nullable Int64 so their structural
    # nulls survive (DNFs in position, pre-2014 drivers in number).
    dfs['results']['position'] = dfs['results']['position'].astype('Int64')
    dfs['drivers']['number'] = dfs['drivers']['number'].astype('Int64')

    # Reproducibility check: duration_seconds is fully populated.
    # This is the guard that catches the original parse bug if it ever
    # returns. milliseconds had zero nulls, so the division must too.
    assert dfs['pit_stops']['duration_seconds'].isna().sum() == 0, \
        "duration_seconds has nulls, the milliseconds parse regressed"

    # Reproducibility check: the Int64 conversion actually took.
    assert dfs['results']['position'].dtype == 'Int64', \
        "position is not nullable Int64"
    assert dfs['drivers']['number'].dtype == 'Int64', \
        "number is not nullable Int64"

    return dfs


def export_clean(dfs, out_dir='../data/clean'):
    """Write each cleaned table to CSV in the clean/ folder.

    CSV does not preserve the nullable Int64 dtype, so downstream
    notebooks re-cast position and number on load. That is expected
    and documented in notebook 01.
    """
    for name, df in dfs.items():
        df.to_csv(f'{out_dir}/{name}.csv', index=False)

    # Reproducibility check: every expected file now exists on disk
    for name in dfs:
        expected_path = f'{out_dir}/{name}.csv'
        assert os.path.exists(expected_path), \
            f"Expected output file was not written: {expected_path}"

    return dfs


def main():
    """Run the full cleaning pipeline in order."""
    dfs = load_raw()
    dfs = filter_years(dfs)
    dfs = replace_null_placeholders(dfs)
    dfs = trim_columns(dfs)
    dfs = rename_columns(dfs)
    dfs = fix_dtypes(dfs)
    dfs = export_clean(dfs)
    print("Cleaning pipeline finished. Clean files written to data/clean.")
    return dfs


if __name__ == '__main__':
    main()