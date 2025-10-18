import pandas as pd
import numpy as np

class FeatureGenerator:
    # String constants for key and template names
    LINEAR_INTERACTIONS = "linear_interactions"
    POLY_INTERACTIONS = "polynomial_interactions"
    EXPLICIT_PAIRWISE_INTERACTIONS = "explicit_pairwise_interactions"
    LINEAR_FEAT_TMPL = "{}_X_{}_linear"
    POLY_DIV_TMPL = "{}_DIV_{}"
    PAIRWISE_FEAT_TMPL = "{}_X_{}_pairwise"
    DIV_SEP = "/"
    NO_CONFIG_MSG = "No feature generation configuration found, returning original dataset"
    CREATED_LINEAR_MSG = "Created linear interaction feature: {} from {} * {}"
    CREATED_POLY_MSG = "Created polynomial division feature: {} from {} / {}"
    CREATED_PAIRWISE_MSG = "Created explicit pairwise feature: {} from {} * {}"
    GEN_DONE_MSG = "Feature generation completed. Generated {} new features. Total features: {}"
    INTERACTION_PAIR_MSG = "Processing {} linear interaction pairs"
    POLY_DIV_PAIR_MSG = "Processing {} polynomial division interactions"
    PAIRWISE_MULT_PAIR_MSG = "Processing {} explicit pairwise multiplication interactions"

    def __init__(self, config):
        self.config = config
        self.feature_generation = config.get_feature_generation()
        print(f"FeatureGenerator initialized with configuration containing {len(self.feature_generation)} generation rules")
    
    def generate(self, df, target):
        print(f"Starting feature generation on dataset with {df.shape[1]} columns")
        df_copy = df.copy()
        
        feature_cols = [col for col in df_copy.columns if col != target]
        print(f"Available feature columns for interaction: {feature_cols}")
        
        if not self.feature_generation:
            print(self.NO_CONFIG_MSG)
            return df_copy
        
        original_feature_count = df_copy.shape[1]
        
        if self.LINEAR_INTERACTIONS in self.feature_generation:
            linear_interactions = self.feature_generation[self.LINEAR_INTERACTIONS]
            print(self.INTERACTION_PAIR_MSG.format(len(linear_interactions)))
            for interaction in linear_interactions:
                if len(interaction) >= 2:
                    col1, col2 = interaction[0], interaction[1]
                    if col1 in df_copy.columns and col2 in df_copy.columns:
                        new_col_name = self.LINEAR_FEAT_TMPL.format(col1, col2)
                        df_copy[new_col_name] = df_copy[col1] * df_copy[col2]
                        print(self.CREATED_LINEAR_MSG.format(new_col_name, col1, col2))
        
        if self.POLY_INTERACTIONS in self.feature_generation:
            poly_interactions = self.feature_generation[self.POLY_INTERACTIONS]
            print(self.POLY_DIV_PAIR_MSG.format(len(poly_interactions)))
            for interaction in poly_interactions:
                parts = interaction.split(self.DIV_SEP)
                if len(parts) == 2:
                    col1, col2 = parts[0], parts[1]
                    if col1 in df_copy.columns and col2 in df_copy.columns:
                        new_col_name = self.POLY_DIV_TMPL.format(col1, col2)
                        df_copy[new_col_name] = df_copy[col1] / (df_copy[col2] + 1e-10)
                        print(self.CREATED_POLY_MSG.format(new_col_name, col1, col2))
        
        if self.EXPLICIT_PAIRWISE_INTERACTIONS in self.feature_generation:
            pairwise = self.feature_generation[self.EXPLICIT_PAIRWISE_INTERACTIONS]
            print(self.PAIRWISE_MULT_PAIR_MSG.format(len(pairwise)))
            for interaction in pairwise:
                parts = interaction.split(self.DIV_SEP)
                if len(parts) == 2:
                    col1, col2 = parts[0], parts[1]
                    if col1 in df_copy.columns and col2 in df_copy.columns:
                        new_col_name = self.PAIRWISE_FEAT_TMPL.format(col1, col2)
                        df_copy[new_col_name] = df_copy[col1] * df_copy[col2]
                        print(self.CREATED_PAIRWISE_MSG.format(new_col_name, col1, col2))
        
        new_feature_count = df_copy.shape[1]
        generated_features = new_feature_count - original_feature_count
        print(self.GEN_DONE_MSG.format(generated_features, new_feature_count))
        
        return df_copy