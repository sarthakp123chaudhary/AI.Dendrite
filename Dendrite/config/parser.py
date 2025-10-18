import json

class ConfigParser:
    
    def __init__(self, json_path):
        print(f"Loading configuration from: {json_path}")
        with open(json_path, 'r') as f:
            self.config = json.load(f)
        print(f"Configuration loaded successfully with {len(self.config)} top-level keys")
    
    def get_target(self):
        target = self.config.get('design_state_data', {}).get('target', {}).get('target')
        print(f"Extracted target variable: {target}")
        return target
    
    def get_prediction_type(self):
        pred_type = self.config.get('design_state_data', {}).get('target', {}).get('prediction_type')
        print(f"Identified prediction task type: {pred_type}")
        return pred_type
    
    def get_feature_handling(self):
        feature_config = self.config.get('design_state_data', {}).get('feature_handling', {})
        print(f"Retrieved feature handling rules for {len(feature_config)} features")
        return feature_config
    
    def get_feature_generation(self):
        gen_config = self.config.get('design_state_data', {}).get('feature_generation', {})
        print(f"Loaded feature generation settings with {len(gen_config)} configuration items")
        return gen_config
    
    def get_feature_reduction(self):
        reduction_config = self.config.get('design_state_data', {}).get('feature_reduction', {})
        print(f"Parsed feature reduction parameters: {reduction_config}")
        return reduction_config
    
    def get_algorithms(self):
        algo_config = self.config.get('design_state_data', {}).get('algorithms', {})
        print(f"Discovered {len(algo_config)} algorithm configurations in JSON")
        return algo_config
    
    def get_hyperparameters(self):
        hyperparam_config = self.config.get('design_state_data', {}).get('hyperparameters', {})
        print(f"Extracted hyperparameter tuning settings: {len(hyperparam_config)} entries")
        return hyperparam_config