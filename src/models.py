# run the notebook 05 modeling pipeline end to end
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Modeling constants, the single source of truth for this script.
# The five legal features were locked in notebook 05: every feature must
# be knowable BEFORE the race finishes. position_change and teammate_gap
# are banned for leakage (position_change contains the answer,
# teammate_gap is built from finish-derived points at the wrong grain).
FEATURE_COLS = ['grid', 'avg_pit_stop_duration', 'constructorId', 'circuitId', 'era']
TARGET_COL = 'position'
BANNED_COLS = ['position', 'position_change', 'positionText', 'points',
               'raceId', 'driverId', 'statusId', 'year']
RANDOM_STATE = 42

# Known-good numbers from notebook 05, the reproducibility targets.
# Row counts and shapes must match exactly. RMSE values are checked
# within a tolerance instead of exact float equality, because tiny
# floating point differences across sklearn versions are normal.
EXPECTED_ROWS_BEFORE_DROP = 5337
EXPECTED_ROWS_AFTER_DROP = 4990
EXPECTED_FEATURE_COUNT = 64
EXPECTED_TRAIN_ROWS = 3992
EXPECTED_TEST_ROWS = 998
EXPECTED_BASELINE_RMSE = 5.405
EXPECTED_SIMPLE_RMSE = 3.347
EXPECTED_TUNED_RMSE = 3.06
RMSE_TOLERANCE = 0.1


def load_processed(processed_dir='../data/processed', clean_dir='../data/clean'):
    """
    Load the tables the model needs: processed results (which already
    carries year and era from features.py), processed avg_pit_stop,
    and clean races (for circuitId, which lives there, not in results).

    era is NOT re-created here. It was stamped once by add_era in
    features.py and travels with results.csv, so the era definition
    cannot drift between the feature script and the model script.
    """
    results = pd.read_csv(f'{processed_dir}/results.csv')
    avg_pit_stop = pd.read_csv(f'{processed_dir}/avg_pit_stop.csv')
    races = pd.read_csv(f'{clean_dir}/races.csv')

    # Reproducibility check: every table loaded and actually has rows,
    # the same hollow-run guard as cleaning.py and features.py.
    for name, df in [('results', results), ('avg_pit_stop', avg_pit_stop),
                     ('races', races)]:
        assert len(df) > 0, f"{name} loaded with zero rows"

    # Reproducibility check: era survived the CSV round-trip. If this
    # fails, features.py was not run or its export broke.
    assert 'era' in results.columns, \
        "results.csv is missing era, run features.py first"
    assert results['era'].isna().sum() == 0, \
        "results.csv has null eras"

    return results, avg_pit_stop, races


def build_modeling_frame(results, avg_pit_stop, races):
    """
    Build the leakage-free modeling frame from notebook 05.

    Start from finished races on a real grid slot (drop DNFs and pit
    lane starts, the same clean rule as the stats notebook), merge in
    avg_pit_stop_duration by team and year, merge in circuitId by race,
    then drop the 347 structurally null pit rows (all 2010, the one
    season this dataset has no pit stop data for). Filling those rows
    would fabricate the exact strategy signal being measured, so they
    are dropped, and dropped BEFORE the split so every model trains
    and tests on the exact same rows.
    """
    # Finished races on a real grid slot
    model_df = results[(results['position'].notna()) & (results['grid'] != 0)].copy()

    # Attach avg_pit_stop_duration by team and year
    model_df = model_df.merge(avg_pit_stop, on=['constructorId', 'year'], how='left')

    # Attach circuitId by race. circuitId lives in races.csv, not results.
    model_df = model_df.merge(races[['raceId', 'circuitId']], on='raceId', how='left')

    # Reproducibility check: the frame before the drop matches the notebook
    assert len(model_df) == EXPECTED_ROWS_BEFORE_DROP, \
        f"Expected {EXPECTED_ROWS_BEFORE_DROP} rows before the drop, got {len(model_df)}"

    # Reproducibility check: every row found its circuit
    assert model_df['circuitId'].isna().sum() == 0, \
        "circuitId has nulls after the merge, orphan races exist"

    # Reproducibility check: the null pit rows are exactly the 2010
    # structural gap and nothing else. If a non-2010 row shows up null,
    # the pit stop merge broke somewhere new.
    null_pit = model_df[model_df['avg_pit_stop_duration'].isna()]
    assert (null_pit['year'] == 2010).all(), \
        "A non-2010 row has a null avg_pit_stop_duration, merge is broken"

    # Drop the structural 2010 rows, before any split
    model_df = model_df.dropna(subset=['avg_pit_stop_duration']).copy()

    # Reproducibility check: the final frame matches the notebook
    assert len(model_df) == EXPECTED_ROWS_AFTER_DROP, \
        f"Expected {EXPECTED_ROWS_AFTER_DROP} rows after the drop, got {len(model_df)}"

    print(f"Modeling frame built: {len(model_df)} rows "
          f"({EXPECTED_ROWS_BEFORE_DROP} before dropping 2010)")

    return model_df


def encode_features(model_df):
    """
    One-hot encode the three categorical features and build X and y.

    constructorId and circuitId are integer IDs (name tags, not
    quantities) and era is text with a real order that the model should
    not assume is evenly spaced. All three get one-hot columns so no
    fake ordering leaks in. After encoding, X is 4,990 rows by 64
    columns (2 numeric + 4 era + 23 constructor + 35 circuit).
    """
    model_encoded = pd.get_dummies(
        model_df,
        columns=['constructorId', 'circuitId', 'era'],
        drop_first=False
    )

    # X starts from the two numeric features, then adds every one-hot column
    base_numeric = ['grid', 'avg_pit_stop_duration']
    encoded_cols = [c for c in model_encoded.columns
                    if c.startswith(('constructorId_', 'circuitId_', 'era_'))]

    X = model_encoded[base_numeric + encoded_cols]
    y = model_encoded[TARGET_COL]

    # Reproducibility check: the leakage firewall. No banned column may
    # appear in X, nothing non-numeric may survive, and both numeric
    # features must be present. A model with leakage produces a
    # beautiful score and is worthless.
    leaked = [c for c in X.columns if c in BANNED_COLS]
    assert leaked == [], f"Banned columns found in X: {leaked}"
    assert not (X.dtypes == 'object').any(), \
        "A non-numeric column survived into X"
    assert 'grid' in X.columns and 'avg_pit_stop_duration' in X.columns, \
        "A numeric feature is missing from X"

    # Reproducibility check: the matrix shape matches the notebook
    assert X.shape == (EXPECTED_ROWS_AFTER_DROP, EXPECTED_FEATURE_COUNT), \
        f"Expected X shape ({EXPECTED_ROWS_AFTER_DROP}, {EXPECTED_FEATURE_COUNT}), got {X.shape}"

    print(f"Feature matrix encoded: X is {X.shape[0]} rows x {X.shape[1]} columns, "
          "leakage firewall passed")

    return X, y


def split_data(X, y):
    """
    Split 80/20 before any model exists, with a fixed random_state so
    the split is reproducible. The test set is held back as unseen
    "new races" and is only used for final scoring.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Reproducibility check: the split sizes match the notebook
    assert len(X_train) == EXPECTED_TRAIN_ROWS, \
        f"Expected {EXPECTED_TRAIN_ROWS} training rows, got {len(X_train)}"
    assert len(X_test) == EXPECTED_TEST_ROWS, \
        f"Expected {EXPECTED_TEST_ROWS} test rows, got {len(X_test)}"

    print(f"Split done: {len(X_train)} train / {len(X_test)} test, "
          f"random_state={RANDOM_STATE}")

    return X_train, X_test, y_train, y_test


def rmse(y_true, y_pred):
    """RMSE in finishing-position units. Lower is better."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def run_baseline(X_train, X_test, y_train, y_test):
    """
    Baseline: a dummy model that always guesses the mean finishing
    position. This is the bar every real model must beat. If a real
    model cannot beat "always guess the average," something is wrong.
    """
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)

    baseline_rmse = rmse(y_test, baseline.predict(X_test))

    # Reproducibility check: the baseline matches the notebook within
    # tolerance. Exact float equality is too strict across versions.
    assert abs(baseline_rmse - EXPECTED_BASELINE_RMSE) < RMSE_TOLERANCE, \
        f"Baseline RMSE {baseline_rmse:.3f} drifted from expected {EXPECTED_BASELINE_RMSE}"

    print(f"Baseline RMSE: {baseline_rmse:.3f} positions "
          f"(always guesses {baseline.constant_[0][0]:.2f})")

    return baseline_rmse


def run_simple_model(X_train, X_test, y_train, y_test):
    """
    Simple model: a shallow Decision Tree on just two features, grid
    (the strongest single predictor) and avg_pit_stop_duration (the
    strategy side). max_depth=5 was chosen by a depth experiment in
    notebook 05: it gave the best test score with almost no overfit
    gap, while an unlimited tree memorized the training data.
    """
    simple_features = ['grid', 'avg_pit_stop_duration']
    X_train_simple = X_train[simple_features]
    X_test_simple = X_test[simple_features]

    simple = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
    simple.fit(X_train_simple, y_train)

    train_rmse = rmse(y_train, simple.predict(X_train_simple))
    test_rmse = rmse(y_test, simple.predict(X_test_simple))

    # Reproducibility check: the simple model matches the notebook
    assert abs(test_rmse - EXPECTED_SIMPLE_RMSE) < RMSE_TOLERANCE, \
        f"Simple model RMSE {test_rmse:.3f} drifted from expected {EXPECTED_SIMPLE_RMSE}"

    print(f"Simple model (Decision Tree, 2 features): "
          f"train RMSE {train_rmse:.3f}, test RMSE {test_rmse:.3f}, "
          f"gap {test_rmse - train_rmse:+.3f}")

    return test_rmse


def select_model_by_cv(X_train, y_train):
    """
    Model selection by 5-fold cross validation on the TRAINING set
    only. Choosing between Random Forest and Gradient Boosting is a
    modeling decision, so it must not touch the test set. The test set
    stays locked for final scoring.
    """
    candidates = {
        'Random Forest': RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
    }

    cv_results = {}
    for name, model in candidates.items():
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        cv_results[name] = -cv_scores.mean()
        print(f"{name}: CV RMSE = {-cv_scores.mean():.3f} "
              f"(+/- {cv_scores.std():.3f})")

    winner = min(cv_results, key=cv_results.get)

    # Reproducibility check: the CV selection lands on the same winner
    # as the notebook. If Random Forest ever wins, the data or the
    # library changed and the notebook conclusions need a re-look.
    assert winner == 'Gradient Boosting', \
        f"CV selected {winner}, notebook 05 selected Gradient Boosting"

    print(f"CV winner: {winner} (selected on training folds only)")

    return winner


def tune_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Tuned model: GridSearchCV over 8 hyperparameter combinations,
    validated by 5-fold cross validation on the training set only.
    The test set is used exactly once, to score the final winner.
    """
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
    }

    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_gb = grid.best_estimator_

    train_rmse = rmse(y_train, best_gb.predict(X_train))
    test_rmse = rmse(y_test, best_gb.predict(X_test))

    # Reproducibility check: the tuned model matches the notebook
    assert abs(test_rmse - EXPECTED_TUNED_RMSE) < RMSE_TOLERANCE, \
        f"Tuned RMSE {test_rmse:.3f} drifted from expected {EXPECTED_TUNED_RMSE}"

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV RMSE (train folds): {-grid.best_score_:.3f}")
    print(f"Tuned Gradient Boosting: train RMSE {train_rmse:.3f}, "
          f"test RMSE {test_rmse:.3f}, gap {test_rmse - train_rmse:+.3f}")

    return best_gb, test_rmse


def report_comparison(baseline_rmse, simple_rmse, tuned_rmse):
    """
    The whole modeling story in one table. Every model was scored on
    the exact same locked test rows, so the RMSE numbers are directly
    comparable. Grid ate most of the meal; the remaining features
    fought over the leftovers.
    """
    comparison = pd.DataFrame({
        'Model': ['Baseline (mean)', 'Simple (Decision Tree, 2 features)',
                  'Tuned (Gradient Boosting, all features)'],
        'Test RMSE': [round(baseline_rmse, 3), round(simple_rmse, 3),
                      round(tuned_rmse, 3)],
        'Gain vs previous': ['-', round(simple_rmse - baseline_rmse, 3),
                             round(tuned_rmse - simple_rmse, 3)],
    })

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    return comparison


def report_feature_importance(best_gb, X_train):
    """
    Sizing the levers: grouped feature importance from the tuned model.

    The model sees 64 columns, but they belong to 5 real-world
    features, because one-hot encoding split constructor, circuit, and
    era into many 0/1 columns. Each one-hot group is summed back to
    its parent so the answer reads in stakeholder language. These
    importances are predictive, not causal, and impurity-based
    importance can favor continuous features like grid (more possible
    split points), both caveats documented in notebook 05.
    """
    importances = pd.Series(best_gb.feature_importances_, index=X_train.columns)

    def group_importance(col_name):
        if col_name.startswith('constructorId_'):
            return 'constructor'
        elif col_name.startswith('circuitId_'):
            return 'circuit'
        elif col_name.startswith('era_'):
            return 'era'
        else:
            return col_name  # grid, avg_pit_stop_duration

    grouped = importances.groupby(group_importance).sum().sort_values(ascending=False)

    # Reproducibility check: importances sum to 1 (they are shares)
    assert abs(grouped.sum() - 1.0) < 0.001, \
        f"Grouped importances sum to {grouped.sum():.4f}, not 1.0"

    # Reproducibility check: grid stays the dominant feature. If it
    # ever loses the top spot, the project's headline finding changed.
    assert grouped.index[0] == 'grid', \
        f"Top feature is {grouped.index[0]}, notebook 05 found grid dominant"

    print("\nFeature importance (grouped, sums to 1.0):")
    for feature, score in grouped.items():
        print(f"  {feature}: {round(score, 3)}")

    return grouped


def residual_check(best_gb, X_test, y_test):
    """
    Sanity check: where does the model make its mistakes?

    Residual = actual minus predicted, bucketed by grid position. The
    mean per bucket is the bias meter (near zero means no systematic
    lean against any part of the field). The std is the chaos meter.
    Notebook 05 found the midfield is the most unpredictable, which
    matters directly to a midfield-team stakeholder.
    """
    residuals = y_test - best_gb.predict(X_test)

    resid_df = pd.DataFrame({
        'grid': X_test['grid'],
        'residual': residuals,
    })

    resid_df['grid_bucket'] = pd.cut(
        resid_df['grid'],
        bins=[0, 5, 10, 15, 25],
        labels=['P1-P5', 'P6-P10', 'P11-P15', 'P16+'],
    )

    bucket_stats = (resid_df.groupby('grid_bucket', observed=True)['residual']
                    .agg(['mean', 'std', 'count']).round(3))

    # Reproducibility check: no bucket carries a large systematic bias.
    # Notebook 05 found all bucket means near zero (largest +0.339), so
    # a mean beyond +/- 1 position would signal something broke.
    assert bucket_stats['mean'].abs().max() < 1.0, \
        "A grid bucket has a mean residual beyond one position, bias appeared"

    print("\nResidual check by grid bucket:")
    print(bucket_stats.to_string())

    return bucket_stats


def main():
    """Run the full modeling pipeline in order."""
    results, avg_pit_stop, races = load_processed()

    model_df = build_modeling_frame(results, avg_pit_stop, races)
    X, y = encode_features(model_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline_rmse = run_baseline(X_train, X_test, y_train, y_test)
    simple_rmse = run_simple_model(X_train, X_test, y_train, y_test)
    select_model_by_cv(X_train, y_train)
    best_gb, tuned_rmse = tune_gradient_boosting(X_train, X_test, y_train, y_test)

    report_comparison(baseline_rmse, simple_rmse, tuned_rmse)
    report_feature_importance(best_gb, X_train)
    residual_check(best_gb, X_test, y_test)

    print("\nModeling pipeline finished. Final model: tuned Gradient Boosting.")
    return best_gb


if __name__ == '__main__':
    main()