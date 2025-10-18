from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ModelFactory:
    
    CLASSIFICATION_MODELS = {
        'LogisticRegression': LogisticRegression,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'GBTClassifier': GradientBoostingClassifier,
        'SVC': SVC,
        'SVM': SVC,
        'KNeighborsClassifier': KNeighborsClassifier,
        'KNN': KNeighborsClassifier,
        'GaussianNB': GaussianNB,
    }
    
    REGRESSION_MODELS = {
        'LinearRegression': LinearRegression,
        'Ridge': Ridge,
        'RidgeRegression': Ridge,
        'Lasso': Lasso,
        'LassoRegression': Lasso,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'GBTRegressor': GradientBoostingRegressor,
        'SVR': SVR,
        'KNeighborsRegressor': KNeighborsRegressor,
    }
    
    def __init__(self, config):
        self.config = config
        self.algorithms = config.get_algorithms()
        print(f"ModelFactory initialized with {len(self.algorithms)} algorithm configurations")
        print(f"Available classification models: {list(self.CLASSIFICATION_MODELS.keys())}")
        print(f"Available regression models: {list(self.REGRESSION_MODELS.keys())}")
    
    def get_models(self, prediction_type):
        print(f"Building model registry for {prediction_type} task")
        models = {}
        
        if prediction_type == 'Classification':
            model_map = self.CLASSIFICATION_MODELS
            print("Using classification model mapping")
        else:
            model_map = self.REGRESSION_MODELS
            print("Using regression model mapping")
        
        for algo_key, algo_config in self.algorithms.items():
            if not isinstance(algo_config, dict):
                print(f"Skipping {algo_key} - invalid configuration format")
                continue
            
            if not algo_config.get('is_selected', False):
                print(f"Skipping {algo_key} - not selected in configuration")
                continue
            
            model_class = None
            
            if algo_key in model_map:
                model_class = model_map[algo_key]
                print(f"Found direct match for {algo_key}")
            elif 'model_name' in algo_config:
                model_name = algo_config['model_name']
                if model_name in model_map:
                    model_class = model_map[model_name]
                    print(f"Found model by name: {model_name} for {algo_key}")
                elif model_name.replace(' ', '') in model_map:
                    model_class = model_map[model_name.replace(' ', '')]
                    print(f"Found model by normalized name: {model_name.replace(' ', '')} for {algo_key}")
            
            if model_class:
                param_grid = self._extract_param_grid(algo_key, algo_config, model_class)
                models[algo_key] = {
                    'class': model_class,
                    'params': param_grid
                }
                print(f"Successfully registered model: {algo_key} with {len(param_grid) if param_grid else 0} hyperparameter groups")
            else:
                print(f"WARNING: Could not find model class for {algo_key} in {prediction_type} models")
        
        print(f"Model registry completed with {len(models)} active models")
        return models
    
    def _extract_param_grid(self, algo_key, algo_config, model_class):
        print(f"Extracting hyperparameter grid for {algo_key}")
        param_grid = {}
        
        if algo_key == 'RandomForestRegressor' or algo_key == 'RandomForestClassifier':
            print(f"Configuring Random Forest hyperparameters for {algo_key}")
            if 'min_trees' in algo_config and 'max_trees' in algo_config:
                min_trees = algo_config['min_trees']
                max_trees = algo_config['max_trees']
                param_grid['n_estimators'] = list(range(min_trees, max_trees + 1, max((max_trees - min_trees) // 3, 1)))
                print(f"n_estimators range: {min_trees} to {max_trees}")
            
            if 'min_depth' in algo_config and 'max_depth' in algo_config:
                min_depth = algo_config['min_depth']
                max_depth = algo_config['max_depth']
                param_grid['max_depth'] = list(range(min_depth, max_depth + 1, max((max_depth - min_depth) // 2, 1)))
                print(f"max_depth range: {min_depth} to {max_depth}")
            
            if 'min_samples_per_leaf_min_value' in algo_config and 'min_samples_per_leaf_max_value' in algo_config:
                min_samples = algo_config['min_samples_per_leaf_min_value']
                max_samples = algo_config['min_samples_per_leaf_max_value']
                param_grid['min_samples_leaf'] = list(range(min_samples, max_samples + 1, max((max_samples - min_samples) // 2, 1)))
                print(f"min_samples_leaf range: {min_samples} to {max_samples}")
        
        elif algo_key == 'GBTRegressor' or algo_key == 'GBTClassifier':
            print(f"Configuring Gradient Boosting hyperparameters for {algo_key}")
            if 'num_of_BoostingStages' in algo_config:
                stages = algo_config['num_of_BoostingStages']
                if isinstance(stages, list) and len(stages) > 0:
                    param_grid['n_estimators'] = stages
                    print(f"n_estimators options: {stages}")
            
            if 'min_stepsize' in algo_config and 'max_stepsize' in algo_config:
                param_grid['learning_rate'] = np.linspace(
                    algo_config['min_stepsize'], 
                    algo_config['max_stepsize'], 
                    3
                ).tolist()
                print(f"learning_rate range: {algo_config['min_stepsize']} to {algo_config['max_stepsize']}")
            
            if 'min_depth' in algo_config and 'max_depth' in algo_config:
                param_grid['max_depth'] = list(range(
                    algo_config['min_depth'], 
                    algo_config['max_depth'] + 1
                ))
                print(f"max_depth range: {algo_config['min_depth']} to {algo_config['max_depth']}")
            
            if 'min_subsample' in algo_config and 'max_subsample' in algo_config:
                param_grid['subsample'] = np.linspace(
                    algo_config['min_subsample'], 
                    algo_config['max_subsample'], 
                    3
                ).tolist()
                print(f"subsample range: {algo_config['min_subsample']} to {algo_config['max_subsample']}")
        
        elif algo_key == 'DecisionTreeRegressor' or algo_key == 'DecisionTreeClassifier':
            print(f"Configuring Decision Tree hyperparameters for {algo_key}")
            if 'min_depth' in algo_config and 'max_depth' in algo_config:
                param_grid['max_depth'] = list(range(
                    algo_config['min_depth'], 
                    algo_config['max_depth'] + 1
                ))
                print(f"max_depth range: {algo_config['min_depth']} to {algo_config['max_depth']}")
            
            if 'min_samples_per_leaf' in algo_config:
                samples = algo_config['min_samples_per_leaf']
                if isinstance(samples, list):
                    param_grid['min_samples_leaf'] = samples
                    print(f"min_samples_leaf options: {samples}")
            
            criteria = []
            if algo_config.get('use_gini', False):
                criteria.append('gini')
            if algo_config.get('use_entropy', False):
                criteria.append('entropy')
            if criteria:
                param_grid['criterion'] = criteria
                print(f"criterion options: {criteria}")
        
        elif algo_key == 'LinearRegression':
            print(f"Configuring Linear Regression hyperparameters")
            param_grid['fit_intercept'] = [True, False]
        
        elif algo_key == 'RidgeRegression':
            print(f"Configuring Ridge Regression hyperparameters")
            if 'min_regparam' in algo_config and 'max_regparam' in algo_config:
                param_grid['alpha'] = np.linspace(
                    algo_config['min_regparam'], 
                    algo_config['max_regparam'], 
                    5
                ).tolist()
                print(f"alpha (regularization) range: {algo_config['min_regparam']} to {algo_config['max_regparam']}")
        
        elif algo_key == 'LassoRegression':
            print(f"Configuring Lasso Regression hyperparameters")
            if 'min_regparam' in algo_config and 'max_regparam' in algo_config:
                param_grid['alpha'] = np.linspace(
                    algo_config['min_regparam'], 
                    algo_config['max_regparam'], 
                    5
                ).tolist()
                print(f"alpha (regularization) range: {algo_config['min_regparam']} to {algo_config['max_regparam']}")
        
        elif algo_key == 'LogisticRegression':
            print(f"Configuring Logistic Regression hyperparameters")
            if 'min_regparam' in algo_config and 'max_regparam' in algo_config:
                param_grid['C'] = np.linspace(0.1, 2.0, 5).tolist()
                print(f"C (inverse regularization) range: 0.1 to 2.0")
            
            if 'min_iter' in algo_config and 'max_iter' in algo_config:
                param_grid['max_iter'] = [algo_config['min_iter'], algo_config['max_iter']]
                print(f"max_iter options: {algo_config['min_iter']}, {algo_config['max_iter']}")
        
        elif algo_key == 'SVM' or algo_key == 'SVC':
            print(f"Configuring SVM hyperparameters for {algo_key}")
            if 'c_value' in algo_config:
                c_values = algo_config['c_value']
                if isinstance(c_values, list) and len(c_values) > 0:
                    param_grid['C'] = c_values
                    print(f"C values: {c_values}")
            
            kernels = []
            if algo_config.get('linear_kernel', False):
                kernels.append('linear')
            if algo_config.get('polynomial_kernel', False):
                kernels.append('poly')
            if algo_config.get('rep_kernel', False) or algo_config.get('rbf_kernel', False):
                kernels.append('rbf')
            if algo_config.get('sigmoid_kernel', False):
                kernels.append('sigmoid')
            if kernels:
                param_grid['kernel'] = kernels
                print(f"kernel options: {kernels}")
        
        elif algo_key == 'KNN':
            print(f"Configuring K-Nearest Neighbors hyperparameters")
            if 'k_value' in algo_config:
                k_values = algo_config['k_value']
                if isinstance(k_values, list):
                    param_grid['n_neighbors'] = k_values
                    print(f"n_neighbors options: {k_values}")
            
            if algo_config.get('distance_weighting', False):
                param_grid['weights'] = ['uniform', 'distance']
                print(f"weights options: uniform, distance")
        
        if not param_grid:
            print(f"No specific hyperparameters found for {algo_key}, using default configuration")
            if hasattr(model_class(), 'random_state'):
                param_grid['random_state'] = [42]
            else:
                if hasattr(model_class(), 'fit_intercept'):
                    param_grid['fit_intercept'] = [True]
                else:
                    return None
        
        print(f"Parameter grid extraction completed for {algo_key}: {param_grid}")
        return param_grid
    
    def train_model(self, model_config, X, y):
        print(f"Starting model training with data shape: X{X.shape}, y{y.shape}")
        model_class = model_config['class']
        param_grid = model_config['params']
        
        base_model = model_class()
        print(f"Instantiated base model: {model_class.__name__}")
        
        if param_grid is None or not param_grid:
            print("No hyperparameter tuning required - training with default parameters")
            base_model.fit(X, y)
            return base_model, {}, 0.0
        
        print(f"Initiating GridSearchCV with parameter combinations: {param_grid}")
        
        cv_folds = min(5, len(y))
        print(f"Using {cv_folds}-fold cross-validation for hyperparameter optimization")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring=None,
            n_jobs=-1,
            verbose=0
        )
        
        print("Executing grid search hyperparameter optimization...")
        grid_search.fit(X, y)
        
        print(f"Grid search completed. Optimal hyperparameters: {grid_search.best_params_}")
        print(f"Best cross-validation performance score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_