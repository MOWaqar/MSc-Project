"""
AUTHOR: M O WAQAR 20/08/2018

This script prepares 17 datasets for 'Classifier.py' to execute


Output of this script is 17 csv files in 'CodeOutputs' folder
other output like information about number of features (columns) and data points (rows) is printed in the console

General steps for creating each version are as follows:
    1. Create a copy of original dataframe so it is not modified in the process
    2. Perform required transformations
    3. Save as csv in ouput folder with the name assigned by the naming convention
    4. Print file information
    5. Delete unnecessary object to free up space.
"""
 
#Import Packages
#for time logging
import time
from contextlib import contextmanager
#for memory management
import gc
import os
#for custom utility functions
import UtilityFunctions as utils

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

#Set up timer for time logging
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
    
# Mian function executes when this script is run
def main():
    #Set up input and output folders
    input_folder, output_folder = utils.Initialize_Folder_Paths()
    
    #Read dataset from csv data file from'ProjectDataFiles' folder
    
    #if file path or file is not found raise an error stating so
    if(not os.path.exists(input_folder + '\\application_train.csv')):
        print("Input file not found. Make sure input application_train csv file is placed in ProjectDataFiles folder correctly.")
    
    original_df = utils.read_csv_File(input_folder + '\\application_train.csv')
   
    #Replace any outliers. The function is written in Utlilities file with outliers already identified
    original_df = utils.replace_outliers(original_df)

    
    #Remove 'SK_ID_CURR' as it is irrelevent
    original_df = original_df.drop(columns = 'SK_ID_CURR')
    
    #Get list of features along with their types for future use.
    categorical_feats, floatingPoint_feats, bool_feats, integer_feats = utils.identify_feature_types(original_df,
                                                                                                ['TARGET', 'SK_ID_CURR', 'Unnamed :0'])
    """with timer("Prepare File version 1 and its vairants"):
        #Version one with null values included and no categorical features.
        # Copy Original data
        input_df = original_df.copy()
        #Drop features 
        input_df = utils.drop_features(input_df, categorical_feats + bool_feats)  
        
        #Save processed version as csv in the output folder.
        input_df.to_csv(output_folder + '\\DataSetVersion1_a.csv')
        print(('Version 1a file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df
        gc.collect()
        
        #Version one with null values included and categorical features OHE.
        # Copy Original data
        input_df = original_df.copy()
        #OHE function only applied to categorical features with NaN considered as a feature variable
        input_df = pd.get_dummies(input_df, columns=categorical_feats, dtype=np.int64, dummy_na= True)
        
        print(('Version 1b file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        input_df.to_csv(output_folder + '\\DataSetVersion1_b.csv')
        del input_df
        gc.collect()
        
        #Version one with null values included and categorical features WoE encoded.
        input_df = original_df.copy()
        
        #WoE Encoding for categorical features
        for cat_feature in categorical_feats:
            WoE_df = utils.calculate_WOE(input_df, 'TARGET', cat_feature)
            input_df[cat_feature] = input_df[cat_feature].replace(WoE_df.set_index('Value')['WoE'])
        
        print(('Version 1c file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        input_df.to_csv(output_folder + '\\DataSetVersion1_c.csv')
        del input_df, WoE_df
        gc.collect()
        
    with timer("Process File version 2 and its variants"):
        #Version with no null values and no categorical features.
        # Copy original
        input_df = original_df.copy()
        
        input_df = utils.drop_features(input_df, categorical_feats + bool_feats)
        
        #In-Built function to drop rows with NaN values
        input_df = input_df.dropna()
        
        input_df.to_csv(output_folder + '\\DataSetVersion2_a.csv')
        print(('Version 2a file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df
        gc.collect()
        
        #Version with no null values and categorical features OHE.
        input_df = original_df.copy()
        
        #In-Built function to drop rows with NaN values
        input_df = input_df.dropna()
        
        #OHE function only applied to categorical features with NaN considered as a feature variable
        input_df = pd.get_dummies(input_df, columns=categorical_feats, dtype=np.int64, dummy_na= True)
        
        print(('Version 2b file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        input_df.to_csv(output_folder + '\\DataSetVersion2_b.csv')
        del input_df
        gc.collect()
        
        #Version with no null values and categorical features WoE Encoded.
        input_df = original_df.copy()
        #In-Built function to drop rows with NaN values
        input_df = input_df.dropna()
        
        #WoE Encoding for categorical features
        for cat_feature in categorical_feats:
            WoE_df = utils.calculate_WOE(input_df, 'TARGET', cat_feature)
            input_df[cat_feature] = input_df[cat_feature].replace(WoE_df.set_index('Value')['WoE'])
        
        print(('Version 2c file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        input_df.to_csv(output_folder + '\\DataSetVersion2_c.csv')
        del input_df, WoE_df
        gc.collect()
    """    
    with timer("Process File version 3 and its varients"):
        #Version with null values imputed and no categorical features.
        input_df = original_df.copy()

        # Drop Categorical and boolean features
        input_df = utils.drop_features(input_df, categorical_feats + bool_feats)
        #Impute missing values
        input_df = utils.impute_features(input_df, 'All')
        print(('Version 3a file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        input_df.to_csv(output_folder + '\\DataSetVersion3_a.csv')
        del input_df
        gc.collect()
        """
        #Nominal variables imputed by mode values and Numerical variables imputed by mean
        # Categoriacal values OHE
        input_df = original_df.copy()
        #Impute missing values   
        input_df = utils.impute_features(input_df, 'All')
        
        #In-Built function to drop rows with NaN values
        input_df = pd.get_dummies(input_df, columns=categorical_feats, dtype=np.int64)
            
        input_df.to_csv(output_folder + '\\DataSetVersion3_b.csv')
        print(('Version 3b file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df
        gc.collect()
        
        #Nominal variables imputed by mode values and Numerical variables imputed by mean
        # Categoriacal values WoE encoded
        input_df = original_df.copy()
        #Impute missing values  
        input_df = utils.impute_features(input_df, 'All')
            
        #Replace categorical columns with WOE columns
        for cat_feature in categorical_feats:
            WoE_df = utils.calculate_WOE(input_df, 'TARGET', cat_feature)
            input_df[cat_feature] = input_df[cat_feature].replace(WoE_df.set_index('Value')['WoE'])
            del WoE_df
            
        input_df.to_csv(output_folder + '\\DataSetVersion3_c.csv')
        print(('Version 3c file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df
        gc.collect()
    with timer("Process File version 4 and its varients"):
        #Nominal variables imputed by mode values and Numerical variables imputed by mean and scaled
        # Categorical features are excluded
        input_df = original_df.copy()
        
        #Drop categorical features
        input_df = utils.drop_features(input_df, categorical_feats + bool_feats)
        
        #Impute missing values
        input_df = utils.impute_features(input_df, 'All')
        
        #Scale numerical features (i.e. integer and floating point features)
        input_df = utils.scale_features(input_df, integer_feats + floatingPoint_feats)
        print(('Version 4a file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        input_df.to_csv(output_folder + '\\DataSetVersion4_a.csv')
        del input_df
        gc.collect()
        
        #Categorical features OHE
        input_df = original_df.copy()
            
        input_df = utils.impute_features(input_df, 'All')
        input_df = utils.scale_features(input_df, integer_feats + floatingPoint_feats)
            
        input_df = pd.get_dummies(input_df, columns=categorical_feats, dtype=np.int64)
            
        input_df.to_csv(output_folder + '\\DataSetVersion4_b.csv')
        print(('Version 4b file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df
        gc.collect()
        
        #Categorical features WoE encoded
        input_df = original_df.copy()
        input_df = utils.impute_features(input_df, 'All')
        input_df = utils.scale_features(input_df, integer_feats + floatingPoint_feats)
            
        #Replace categorical columns with WOE columns
        for cat_feature in categorical_feats:
            WoE_df = utils.calculate_WOE(input_df, 'TARGET', cat_feature)
            input_df[cat_feature] = input_df[cat_feature].replace(WoE_df.set_index('Value')['WoE'])
            del WoE_df
            
        input_df.to_csv(output_folder + '\\DataSetVersion4_c.csv')
        print(('Version 4c file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df
        gc.collect()
        
    with timer("Process File version 5 and its varients"):
        input_df = original_df.copy()
        
        
        
        labels = input_df.pop('TARGET')
        
        input_df = utils.drop_features(input_df, categorical_feats)
        
        feature_df = utils.select_features(input_df, labels, drop_ratio = 0.5)
        
        selected_features = feature_df[feature_df.Total >= 3].Feature.values
        
        input_df = input_df[selected_features]
        input_df = utils.impute_features(input_df, features ='All')    
        input_df = utils.scale_features(input_df)
        
        input_df['TARGET'] = labels 
        input_df.to_csv(output_folder + '\\DataSetVersion5_a.csv')
        print(('Version 5a file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df, selected_features
        gc.collect()
        
        # Categorical features OHE
        input_df = original_df.copy()
        
        #Extract labels as they are required for feature selection function and then add it back at the end.
        labels = input_df.pop('TARGET')
        
        input_df = pd.get_dummies(input_df, columns=categorical_feats, dtype=np.int64)
        
        #Function that runs 5 vote selection algorithm with dropping rougly 50% of features 
        feature_df = utils.select_features(input_df, labels, drop_ratio = 0.5)
        
        #Short list only those features that get more than 3 votes
        selected_features = feature_df[feature_df.Total >= 3].Feature.values
        input_df = input_df[selected_features]
        
        input_df = utils.impute_features(input_df, features ='All')    
        input_df = utils.scale_features(input_df)
        
        input_df['TARGET'] = labels 
        input_df.to_csv(output_folder + '\\DataSetVersion5_b.csv')
        print(('Version 5b file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df, selected_features
        gc.collect()
        
        # Categoriacal features WoE encoded
        input_df = original_df.copy()
                
        #Replace categorical columns with WOE columns
        for cat_feature in categorical_feats:
            WoE_df = utils.calculate_WOE(input_df, 'TARGET', cat_feature)
            input_df[cat_feature] = input_df[cat_feature].replace(WoE_df.set_index('Value')['WoE'])
            del WoE_df
            
        labels = input_df.pop('TARGET')
        feature_df = utils.select_features(input_df, labels, drop_ratio = 0.5)
        
        selected_features = feature_df[feature_df.Total >= 3].Feature.values
        input_df = input_df[selected_features]
        
        input_df = utils.impute_features(input_df, features ='All')    
        input_df = utils.scale_features(input_df)
        
        input_df['TARGET'] = labels 
        input_df.to_csv(output_folder + '\\DataSetVersion5_c.csv')
        print(('Version 5c file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
        del input_df
        gc.collect()
     """   
    with timer("Process File version 6 and its varients"):
        # Balanced sample with categorical feature OHE
        input_df = original_df.copy()
        
        labels = input_df.pop('TARGET')
        
        input_df = pd.get_dummies(input_df, columns=categorical_feats, dtype=np.int64)
        
        input_df = utils.impute_features(input_df, features ='All')    
        input_df = utils.scale_features(input_df)
        
        sampled_df , sampled_labels = utils.take_sample(input_df, labels, stratified = False)
        sampled_df['TARGET'] = sampled_labels
        sampled_df.to_csv(output_folder + '\\DataSetVersion6_a.csv')
        print(('Version 6a file contains {} feature and {} data instances.\n').format(sampled_df.shape[1], sampled_df.shape[0]))
        del input_df, sampled_df, labels
        gc.collect()
        
        # Stratified sample with categorical feature OHE
        input_df = original_df.copy()
        
        labels = input_df.pop('TARGET')
        
        input_df = pd.get_dummies(input_df, columns=categorical_feats, dtype=np.int64)
        
        input_df = utils.impute_features(input_df, features ='All')    
        input_df = utils.scale_features(input_df)
        
        sampled_df , sampled_labels = utils.take_sample(input_df, labels)
        sampled_df['TARGET'] = sampled_labels
        sampled_df.to_csv(output_folder + '\\DataSetVersion6_b.csv')
        print(('Version 6b file contains {} feature and {} data instances.\n').format(sampled_df.shape[1], sampled_df.shape[0]))
        del input_df, sampled_df, labels
        gc.collect()
    
    


if __name__ == "__main__":
    with timer("Data preperation code"):
        main()

