#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from typing import List, Tuple


# In[11]:


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new, potentially more predictive features from the existing data.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new features added.
    """
    # Creating a copy
    df_engineered = df.copy()

    # Creating the FEV1/FVC ratio, a key clinical metric
    # Adding a small epsilon to prevent division by zero
    df_engineered['FEV1_FVC_Ratio'] = df_engineered['LungFunctionFEV1'] / (df_engineered['LungFunctionFVC'] + 1e-6)

    # Creating an aggregated symptom score
    symptom_columns = ['Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms', 'ExerciseInduced']
    df_engineered['SymptomScore'] = df_engineered[symptom_columns].sum(axis=1)

    return df_engineered

def select_and_drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the final features for the model by dropping irrelevant,
    redundant, or weak predictor columns.

    Args:
        df (pd.DataFrame): The DataFrame with engineered features.

    Returns:
        pd.DataFrame: The DataFrame with only the selected features.
    """
    # Original symptom columns are now redundant
    symptom_columns = ['Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms', 'ExerciseInduced']

    # Columns to drop based on EDA and feature engineering
    columns_to_drop = [
        'PatientID', 'DoctorInCharge', # Irrelevant identifiers
        'Age', 'Gender', 'Smoking',      # Weak predictors identified in EDA
    ] + symptom_columns

    return df.drop(columns=columns_to_drop)

def process_data(raw_df: pd.DataFrame, target_column: str = 'Diagnosis', test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Main function to run the entire data preprocessing and manipulation pipeline.

    Args:
        raw_df (pd.DataFrame): The raw input DataFrame.
        target_column (str): The name of the target variable column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed for reproducibility.

    Returns:
        Tuple: A tuple containing:
            - X_train_resampled: The balanced and preprocessed training features.
            - y_train_resampled: The balanced training labels.
            - X_test_processed: The preprocessed testing features.
            - y_test: The original, untouched testing labels.
            - preprocessor: The fitted ColumnTransformer object.
    """
    # Feature engineering and selection
    df_engineered = engineer_features(raw_df)
    df_selected = select_and_drop_features(df_engineered)

    # Defining Features (X) and Target (y)
    X = df_selected.drop(columns=[target_column])
    y = df_selected[target_column]

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Identifying feature types for the pipeline
    categorical_features = X.select_dtypes(include=['int64', 'object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['float64']).columns.tolist()

    # Creating and build the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Fitting on training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Handling class imbalance using SMOTE on the training data
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print("Data processing complete.")
    return X_train_resampled, y_train_resampled, X_test_processed, y_test, preprocessor


# In[ ]:




