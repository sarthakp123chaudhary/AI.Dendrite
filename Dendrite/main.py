import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.parser import ConfigParser
from pipeline.preprocessing import DataPreprocessor
from pipeline.feature_generation import FeatureGenerator
from pipeline.feature_reduction import FeatureReducer
from pipeline.model_factory import ModelFactory
from utils.metrics import MetricsLogger

def main(json_path, csv_path):
    
    print("INITIALIZING MACHINE LEARNING PIPELINE EXECUTION")
    print(f"Configuration file: {json_path}")
    print(f"Dataset file: {csv_path}")
    
    config = ConfigParser(json_path)
    target = config.get_target()
    prediction_type = config.get_prediction_type()
    print(f"Pipeline configured for target variable: {target}")
    print(f"Machine learning task identified as: {prediction_type}")
    
    print("LOADING AND VALIDATING DATASET")
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded with dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Column names detected: {list(df.columns)}")
    
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in dataset columns: {df.columns.tolist()}")
    
    print("EXECUTING DATA PREPROCESSING PIPELINE")
    preprocessor = DataPreprocessor(config)
    df_processed = preprocessor.fit_transform(df, target)
    print(f"Preprocessing completed. Dataset shape after preprocessing: {df_processed.shape}")
    
    print("EXECUTING FEATURE GENERATION PIPELINE")
    feature_generator = FeatureGenerator(config)
    df_with_features = feature_generator.generate(df_processed, target)
    print(f"Feature generation completed. Dataset shape after feature creation: {df_with_features.shape}")
    
    print("EXECUTING FEATURE REDUCTION PIPELINE")
    feature_reducer = FeatureReducer(config)
    X_reduced, y = feature_reducer.fit_transform(df_with_features, target, prediction_type)
    print(f"Feature reduction completed. Final feature matrix dimensions: {X_reduced.shape}")
    print(f"Target vector dimensions: {y.shape}")
    
    print("INITIALIZING MODEL FACTORY AND TRAINING PIPELINE")
    model_factory = ModelFactory(config)
    models = model_factory.get_models(prediction_type)
    
    print(f"Model factory configured with {len(models)} algorithms for training")
    
    metrics_logger = MetricsLogger(prediction_type)
    
    for model_name, model_config in models.items():
        
        print(f"TRAINING MODEL: {model_name}")
        print(f"Model class: {model_config['class'].__name__}")
        
        best_model, best_params, best_score = model_factory.train_model(
            model_config, X_reduced, y
        )
        
        print(f"Generating predictions using optimized {model_name}")
        y_pred = best_model.predict(X_reduced)
        
        metrics_logger.log_metrics(model_name, y, y_pred, best_params, best_score)
    
    print("MACHINE LEARNING PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print(f"Total models trained and evaluated: {len(models)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        
        sys.exit(1)
    
    json_path = sys.argv[1]
    csv_path = sys.argv[2]
    
    try:
        main(json_path, csv_path)
    except Exception as e:
        print(f"\nPIPELINE EXECUTION FAILED: {str(e)}")
        import traceback
        print("DETAILED ERROR TRACEBACK:")
        traceback.print_exc()
        sys.exit(1)