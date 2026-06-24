import pytest
import numpy as np
import pandas as pd
from Aura.utils.cleaning import StatefulCleaner

def test_impute_mean_median_mode():
    # Construct a dataset with missing values
    data = {
        "num_val": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat_val": ["A", "B", "A", np.nan, "A"]
    }
    df = pd.DataFrame(data)
    
    actions = [
        {"column": "num_val", "actionType": "impute_mean"},
        {"column": "cat_val", "actionType": "impute_mode"}
    ]
    
    cleaner = StatefulCleaner(actions)
    cleaner.fit(df)
    
    # Check fitted values
    assert cleaner.imputers["num_val"] == 3.0  # Mean of 1, 2, 4, 5 is 3
    assert cleaner.imputers["cat_val"] == "A"   # Mode is A
    
    transformed = cleaner.transform(df)
    assert transformed["num_val"].isna().sum() == 0
    assert transformed.loc[2, "num_val"] == 3.0
    assert transformed["cat_val"].isna().sum() == 0
    assert transformed.loc[3, "cat_val"] == "A"

def test_impute_median():
    data = {
        "num_val": [1.0, 2.0, np.nan, 10.0, 20.0]
    }
    df = pd.DataFrame(data)
    
    actions = [
        {"column": "num_val", "actionType": "impute_median"}
    ]
    
    cleaner = StatefulCleaner(actions)
    cleaner.fit(df)
    
    assert cleaner.imputers["num_val"] == 6.0  # Median of 1, 2, 10, 20 is 6
    
    transformed = cleaner.transform(df)
    assert transformed.loc[2, "num_val"] == 6.0

def test_impute_knn_and_mice():
    data = {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    }
    df = pd.DataFrame(data)
    
    actions_knn = [
        {"column": "x", "actionType": "impute_knn"}
    ]
    cleaner_knn = StatefulCleaner(actions_knn)
    cleaner_knn.fit(df)
    transformed_knn = cleaner_knn.transform(df)
    assert transformed_knn["x"].isna().sum() == 0
    
    actions_mice = [
        {"column": "x", "actionType": "impute_mice"}
    ]
    cleaner_mice = StatefulCleaner(actions_mice)
    cleaner_mice.fit(df)
    transformed_mice = cleaner_mice.transform(df)
    assert transformed_mice["x"].isna().sum() == 0

def test_outliers_clip_drop():
    # Clip outliers
    data = {
        "val": [1.0, 1.1, 1.2, 1.3, 100.0]
    }
    df = pd.DataFrame(data)
    
    actions_clip = [
        {"column": "val", "actionType": "clip_outliers"}
    ]
    cleaner_clip = StatefulCleaner(actions_clip)
    cleaner_clip.fit(df)
    transformed_clip = cleaner_clip.transform(df)
    assert transformed_clip["val"].max() < 100.0
    
    # Drop outliers (only in training)
    actions_drop = [
        {"column": "val", "actionType": "drop_outliers"}
    ]
    cleaner_drop = StatefulCleaner(actions_drop)
    cleaner_drop.fit(df)
    
    # In training, the outlier 100.0 is dropped
    transformed_train = cleaner_drop.transform(df, is_training=True)
    assert len(transformed_train) == 4
    
    # Not in training, the outlier is preserved
    transformed_test = cleaner_drop.transform(df, is_training=False)
    assert len(transformed_test) == 5

def test_isolation_forest():
    # 50 normal values and 2 extreme outliers
    np.random.seed(42)
    vals = np.random.normal(loc=10.0, scale=1.0, size=50).tolist()
    vals.extend([1000.0, -1000.0])
    df = pd.DataFrame({"val": vals})
    
    actions = [
        {"column": "val", "actionType": "isolation_forest"}
    ]
    cleaner = StatefulCleaner(actions)
    cleaner.fit(df)
    
    # Outliers should be dropped in training transform
    transformed_train = cleaner.transform(df, is_training=True)
    assert len(transformed_train) < 52
    assert 1000.0 not in transformed_train["val"].values
    
    transformed_test = cleaner.transform(df, is_training=False)
    assert len(transformed_test) == 52

def test_drop_column():
    data = {
        "x": [1, 2, 3],
        "y": [4, 5, 6]
    }
    df = pd.DataFrame(data)
    actions = [
        {"column": "y", "actionType": "drop"}
    ]
    cleaner = StatefulCleaner(actions)
    cleaner.fit(df)
    transformed = cleaner.transform(df)
    assert "y" not in transformed.columns
    assert "x" in transformed.columns

def test_new_transformations():
    data = {
        "num_val": [1.0, 2.0, 3.0],
        "other_num": [2.0, 3.0, 4.0],
        "date_val": ["2026-06-23 18:00:00", "2026-06-24 19:30:00", "2026-06-25 20:00:00"]
    }
    df = pd.DataFrame(data)
    
    actions = [
        {"column": "num_val", "actionType": "transform_log"},
        {"column": "num_val", "actionType": "transform_power"},
        {"column": "num_val", "actionType": "transform_interaction:other_num"},
        {"column": "date_val", "actionType": "transform_date"}
    ]
    
    cleaner = StatefulCleaner(actions)
    cleaner.fit(df)
    transformed = cleaner.transform(df)
    
    assert "num_val_log" in transformed.columns
    assert "num_val_power" in transformed.columns
    assert "num_val_x_other_num" in transformed.columns
    assert "date_val_year" in transformed.columns
    assert "date_val_month" in transformed.columns
    
    # Check log transform values
    assert np.allclose(transformed["num_val_log"], np.log1p(df["num_val"]))
    # Check power transform values
    assert np.allclose(transformed["num_val_power"], np.square(df["num_val"]))
    # Check interaction transform values
    assert np.allclose(transformed["num_val_x_other_num"], df["num_val"] * df["other_num"])
    # Check date part extraction values
    assert transformed.loc[0, "date_val_year"] == 2026
    assert transformed.loc[0, "date_val_month"] == 6
    assert transformed.loc[0, "date_val_day"] == 23

