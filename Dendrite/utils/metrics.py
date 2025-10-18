import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

class MetricsLogger:
    
    def __init__(self, prediction_type):
        self.prediction_type = prediction_type
        print(f"MetricsLogger initialized for {prediction_type} evaluation")
    
    def log_metrics(self, model_name, y_true, y_pred, best_params, best_cv_score):
        print(f"\n{'='*60}")
        print(f"PERFORMANCE EVALUATION RESULTS FOR: {model_name}")
        print(f"{'='*60}")
        
        print(f"\nOptimal Hyperparameter Configuration:")
        if best_params:
            for param, value in best_params.items():
                print(f"  {param}: {value}")
        else:
            print("  No hyperparameter tuning performed (default settings used)")
        
        print(f"\nCross-Validation Performance Score: {best_cv_score:.4f}")
        
        if self.prediction_type == 'Classification':
            self._log_classification_metrics(y_true, y_pred)
        else:
            self._log_regression_metrics(y_true, y_pred)
    
    def _log_classification_metrics(self, y_true, y_pred):
        print("\nClassification Performance Metrics:")
        print("-" * 40)
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Overall Classification Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        unique_classes = len(np.unique(y_true))
        print(f"Number of distinct classes in dataset: {unique_classes}")
        
        try:
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            print(f"Macro-averaged Precision Score: {precision:.4f}")
            print(f"Macro-averaged Recall Score: {recall:.4f}")
            print(f"Macro-averaged F1-Score: {f1:.4f}")
        except Exception as e:
            print(f"Error computing precision/recall/f1 metrics: {e}")
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            print(f"\nConfusion Matrix Analysis:")
            print("Predicted vs Actual class distribution:")
            print(cm)
            
            print(f"Confusion matrix shape: {cm.shape}")
            total_predictions = np.sum(cm)
            correct_predictions = np.trace(cm)
            print(f"Total predictions: {total_predictions}, Correct: {correct_predictions}")
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
    
    def _log_regression_metrics(self, y_true, y_pred):
        print("\nRegression Performance Metrics:")
        print("-" * 40)
        
        mse = mean_squared_error(y_true, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        mae = mean_absolute_error(y_true, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        r2 = r2_score(y_true, y_pred)
        print(f"Coefficient of Determination (R²): {r2:.4f}")
        
        if r2 >= 0:
            print(f"Model explains {r2*100:.2f}% of target variable variance")
        else:
            print(f"Model performs worse than baseline (negative R²: {r2:.4f})")
        
        n = len(y_true)
        if n > 2:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
            print(f"Adjusted R² Score (sample size corrected): {adj_r2:.4f}")
        
        target_range = np.max(y_true) - np.min(y_true)
        print(f"Target variable range: {np.min(y_true):.4f} to {np.max(y_true):.4f} (span: {target_range:.4f})")
        print(f"RMSE as percentage of target range: {(rmse/target_range)*100:.2f}%")