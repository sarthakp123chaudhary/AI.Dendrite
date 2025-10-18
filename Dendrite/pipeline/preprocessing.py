import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    
    def __init__(self, config):
        self.config = config
        self.feature_handling = config.get_feature_handling()
        self.imputers = {}
        self.label_encoders = {}
        print(f"DataPreprocessor initialized with {len(self.feature_handling)} feature configurations")
    
    def fit_transform(self, df, target):
        print(f"Starting preprocessing on dataset with shape: {df.shape}")
        df_copy = df.copy()
        
        feature_cols = [col for col in df_copy.columns if col != target]
        print(f"Processing {len(feature_cols)} feature columns (excluding target: {target})")
        
        for col in feature_cols:
            if col in self.feature_handling:
                feature_config = self.feature_handling[col]
                
                if not feature_config.get('is_selected', True):
                    print(f"Dropping unselected feature: {col}")
                    df_copy = df_copy.drop(columns=[col])
                    continue
                
                feature_variable_type = feature_config.get('feature_variable_type', 'numerical')
                print(f"Processing {col} as {feature_variable_type} type")
                
                if feature_variable_type in ['text', 'categorical']:
                    print(f"Applying label encoding transformation to categorical feature: {col}")
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"Label encoder fitted for {col} with {len(le.classes_)} unique classes")
                    continue
                
                impute_method = feature_config.get('feature_details', {}).get('missing_values', 'Impute')
                
                if impute_method != 'Impute':
                    print(f"Skipping imputation for {col} (method: {impute_method})")
                    continue
                
                impute_with = feature_config.get('feature_details', {}).get('impute_with', 'Average of values')
                print(f"Applying {impute_with} imputation strategy to numerical feature: {col}")
                
                impute_strategies = {
                    'Average of values': ('mean', None, f"Using mean imputation for {col}"),
                    'Median of values': ('median', None, f"Using median imputation for {col}"),
                    'Mode of values': ('most_frequent', None, f"Using mode imputation for {col}"),
                    'custom': ('constant', feature_config.get('feature_details', {}).get('impute_value', 0), None),
                    'custom value': ('constant', feature_config.get('feature_details', {}).get('impute_value', 0), None),
                }

                if impute_with in impute_strategies:
                    strategy, fill_value, msg = impute_strategies[impute_with]
                    if strategy == 'constant':
                        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                        print(f"Using custom value {fill_value} for imputing {col}")
                    else:
                        imputer = SimpleImputer(strategy=strategy)
                        if msg: print(msg)
                else:
                    imputer = SimpleImputer(strategy='mean')
                    print(f"Defaulting to mean imputation for {col}")
                
                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                self.imputers[col] = imputer
                print(f"Imputation completed for {col}")
        
        print(f"Preprocessing completed. Final dataset shape: {df_copy.shape}")
        return df_copy