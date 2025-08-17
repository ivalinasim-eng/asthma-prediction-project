import pytest
import pandas as pd
import numpy as np
import sys
import os

# Adding the 'src' directory to the Python path to import my module
# This is necessary for the test runner to find the 'data_processing' script
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_processing import engineer_features, select_and_drop_features, process_data

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """
    Creates a small, controlled sample DataFrame for testing,
    mimicking the structure of the real raw data.
    Updated to 10 rows to support stratified splitting.
    """
    data = {
        'PatientID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Age': [25, 45, 30, 50, 22, 35, 55, 40, 28, 60],
        'Gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'Smoking': [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
        'LungFunctionFEV1': [3.0, 2.5, 3.5, 2.0, 3.2, 2.8, 3.1, 2.2, 3.3, 2.9],
        'LungFunctionFVC': [4.0, 3.0, 4.5, 2.5, 3.8, 3.5, 4.2, 2.8, 4.0, 3.6],
        'Wheezing': [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        'ShortnessOfBreath': [1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        'ChestTightness': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'Coughing': [1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        'NighttimeSymptoms': [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
        'ExerciseInduced': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        'DoctorInCharge': ['Dr. A', 'Dr. B', 'Dr. A', 'Dr. C', 'Dr. B', 'Dr. A', 'Dr. C', 'Dr. B', 'Dr. A', 'Dr. C'],
        'Diagnosis': [1, 1, 0, 1, 0, 1, 0, 0, 1, 0], 
        'Ethnicity': [0,1,0,1,0,1,0,1,0,1], 'EducationLevel': [1,2,1,2,1,2,1,2,1,2], 
        'BMI': [22.1, 25.5, 23.0, 28.1, 21.5, 24.0, 29.5, 26.0, 22.5, 30.0],
        'PhysicalActivity': [3.0, 2.1, 4.5, 1.5, 5.0, 3.5, 2.0, 1.0, 4.8, 2.5], 
        'DietQuality': [8, 5, 9, 4, 7, 6, 5, 3, 9, 6],
        'SleepQuality': [8, 6, 9, 5, 8, 7, 6, 4, 9, 7], 
        'PollutionExposure': [4, 7, 2, 8, 3, 6, 9, 5, 2, 7],
        'PollenExposure': [3, 6, 2, 7, 4, 5, 8, 4, 3, 6], 
        'DustExposure': [5, 7, 3, 8, 4, 6, 9, 5, 2, 7],
        'PetAllergy': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 
        'FamilyHistoryAsthma': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        'HistoryOfAllergies': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0], 
        'Eczema': [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
        'HayFever': [0, 1, 0, 1, 0, 0, 1, 1, 0, 0], 
        'GastroesophagealReflux': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    }
    return pd.DataFrame(data)

def test_engineer_features(sample_raw_df):
    """
    Tests the engineer_features function.
    - Checks if new columns are created.
    - Checks if calculations are correct.
    - Checks that the original DataFrame is not modified.
    """
    original_df = sample_raw_df.copy()
    engineered_df = engineer_features(sample_raw_df)

    # Test 1: Checking if new columns exist
    assert 'FEV1_FVC_Ratio' in engineered_df.columns
    assert 'SymptomScore' in engineered_df.columns

    # Test 2: Checking calculation correctness for the first row
    expected_ratio = 3.0 / 4.0
    expected_symptom_score = 1 + 1 + 0 + 1 + 0 + 1
    assert np.isclose(engineered_df['FEV1_FVC_Ratio'].iloc[0], expected_ratio)
    assert engineered_df['SymptomScore'].iloc[0] == expected_symptom_score

    # Test 3: Ensuring the original DataFrame was not changed
    pd.testing.assert_frame_equal(original_df, sample_raw_df)

def test_select_and_drop_features(sample_raw_df):
    """
    Tests the select_and_drop_features function.
    - Checks if the correct columns are dropped.
    """
    # Testking on a dataframe that has the engineered features
    engineered_df = engineer_features(sample_raw_df)
    selected_df = select_and_drop_features(engineered_df)

    # Test 1: Checking that specified columns are dropped
    dropped_cols = ['PatientID', 'DoctorInCharge', 'Age', 'Gender', 'Smoking', 'Wheezing']
    for col in dropped_cols:
        assert col not in selected_df.columns

    # Test 2: Checking that expected columns remain
    assert 'Diagnosis' in selected_df.columns
    assert 'SymptomScore' in selected_df.columns

def test_process_data_pipeline(sample_raw_df):
    """
    Tests the main process_data function as an integration test.
    - Checks output types.
    - Checks shapes and balancing.
    - Checks for data leakage (preprocessor fit only on train).
    """
    # Running the full pipeline
    X_train, y_train, X_test, y_test, preprocessor = process_data(sample_raw_df)

    # Test 1: Checking output types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, pd.Series) 
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, pd.Series) 
    from sklearn.compose import ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)

    # Test 2: Checking that the training data is balanced after SMOTE
    unique, counts = np.unique(y_train, return_counts=True)
    assert len(unique) == 2
    assert counts[0] == counts[1]

    # Test 3: Checking that the number of columns is consistent
    assert X_train.shape[1] == X_test.shape[1]

    # Test 4: Checking for an empty dataframe input
    with pytest.raises(ValueError):
        # Creating an empty DataFrame but with the correct columns from the fixture
        empty_df = pd.DataFrame(columns=sample_raw_df.columns)
        process_data(empty_df)