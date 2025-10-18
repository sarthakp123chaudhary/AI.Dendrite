import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

class FeatureReducer:
    
    def __init__(self, config):
        self.config = config
        self.feature_reduction = config.get_feature_reduction()
        self.reducer = None
        print(f"FeatureReducer initialized with reduction configuration: {self.feature_reduction}")
    
    def fit_transform(self, df, target, prediction_type):
        print(f"Starting feature reduction on dataset with shape: {df.shape}")
        
        X = df.drop(columns=[target])
        y = df[target]
        print(f"Separated features (X: {X.shape}) and target (y: {y.shape})")
        
        reduction_method = self.feature_reduction.get('feature_reduction_method', 'No Reduction')
        num_features = self.feature_reduction.get('num_of_features_to_keep', X.shape[1])
        
        if isinstance(num_features, str):
            num_features = int(num_features)
        
        num_features = min(num_features, X.shape[1])
        
        print(f"Applying reduction method: {reduction_method}")
        print(f"Target features to retain: {num_features} out of {X.shape[1]} original features")
        print(f"Prediction task type: {prediction_type}")
        
        if reduction_method == 'No Reduction':
            print("No reduction applied - returning all original features")
            return X.values, y.values
        
        elif reduction_method == 'Corr with Target':
            print("Computing correlation-based feature selection with target variable")
            correlations = {}
            for col in X.columns:
                correlations[col] = abs(np.corrcoef(X[col], y)[0, 1])
            
            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:num_features]
            selected_cols = [f[0] for f in top_features]
            
            print(f"Top correlated features selected: {selected_cols}")
            for feature, corr in top_features:
                print(f"  {feature}: correlation = {corr:.4f}")
            
            return X[selected_cols].values, y.values
        
        elif reduction_method == 'Tree-based':
            print(f"Applying tree-based feature importance selection for {prediction_type}")
            
            if prediction_type == 'Classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                print("Using RandomForestClassifier for feature importance ranking")
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                print("Using RandomForestRegressor for feature importance ranking")
            
            print("Training tree model to compute feature importances...")
            model.fit(X, y)
            
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_cols = importances.head(num_features)['feature'].tolist()
            print(f"Tree-based feature selection completed. Selected features: {selected_cols}")
            
            for idx, row in importances.head(num_features).iterrows():
                print(f"  {row['feature']}: importance = {row['importance']:.4f}")
            
            return X[selected_cols].values, y.values
        
        elif reduction_method == 'PCA':
            print(f"Applying Principal Component Analysis (PCA) for dimensionality reduction")
            pca = PCA(n_components=num_features, random_state=42)
            
            print(f"Fitting PCA transformation from {X.shape[1]} to {num_features} components")
            X_reduced = pca.fit_transform(X)
            
            print(f"PCA transformation completed. Explained variance per component:")
            for i, var_ratio in enumerate(pca.explained_variance_ratio_):
                print(f"  PC{i+1}: {var_ratio:.4f}")
            
            total_variance = sum(pca.explained_variance_ratio_)
            print(f"Total explained variance retained: {total_variance:.4f} ({total_variance*100:.2f}%)")
            
            self.reducer = pca
            return X_reduced, y.values
        
        else:
            print(f"Unknown reduction method '{reduction_method}' - defaulting to no reduction")
            return X.values, y.values