"""
AUTHOR: M O WAQAR 20/08/2018

Utility Functions used by other scripts are written in this file
It contains code snippets that are taken from different kernel publically shared on Kaggle competition

"""

 

#Import Packages
#memory management
import gc

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# matplotlib for plotting
import matplotlib.pyplot as plt

#to impute and scale
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

#used by feature selection methods
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel, chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

##This method sets up input and output folder paths. 
## NOTE: Folder structure has to be maintained for the code to work.
## Outputs from the code are saved in 'CodeOutputs' folder and and Input is read from 'ProjectDataFiles' folder
## This function would return an error if input folder is not found
def Initialize_Folder_Paths():    
    #Set input data folder 
    dataFolder = os.getcwd() + os.sep + os.pardir + os.sep + 'ProjectDataFiles'
    if(not os.path.exists(dataFolder)):
        print("Input Data folder not found. Please specify data folder path as dataFolder variable to proceed")
        raise NotADirectoryError

    #Create output folder is it does not exist
    outputFolder = os.getcwd() + os.sep + os.pardir + os.sep + 'CodeOutputs'

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        print('Output Folder created')
        
    return dataFolder, outputFolder

##This method reads csv file and returns pandas DataFrame
##Returns an error if file is not found
def read_csv_File(file_path):
    if(not os.path.exists(file_path)):
        raise FileNotFoundError
        
    return pd.read_csv(file_path)

###############Functions used by EDA and pre-processing part

##Function to replace pre determined outliers in input data
def replace_outliers(df):
    if 'CODE_GENDER' in df:
        df = df[df['CODE_GENDER'] != 'XNA']

    #Replace outlier values with nan
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].map(lambda x: x if x <= 0 else np.nan)
    df['REGION_RATING_CLIENT_W_CITY'] = df['REGION_RATING_CLIENT_W_CITY'].map(lambda x: x if x >= 0 else np.nan)
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].map(lambda x: x if x <= 5e6 else np.nan)
    df['AMT_REQ_CREDIT_BUREAU_QRT'] = df['AMT_REQ_CREDIT_BUREAU_QRT'].map(lambda x: x if x <= 10 else np.nan)
    df['OBS_30_CNT_SOCIAL_CIRCLE'] = df['OBS_30_CNT_SOCIAL_CIRCLE'].map(lambda x: x if x <= 40 else np.nan)
    df['OBS_60_CNT_SOCIAL_CIRCLE'] = df['OBS_60_CNT_SOCIAL_CIRCLE'].map(lambda x: x if x <= 50 else np.nan)
    df['DEF_30_CNT_SOCIAL_CIRCLE'] = df['DEF_30_CNT_SOCIAL_CIRCLE'].map(lambda x: x if x <= 100 else np.nan)
    
    return df

## Function to calculate missing values by columns
## Returns a data frame with features name, number of missing values and their %age 
def compute_missing_values(df, sortAscending = False, verbose=True):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=sortAscending).round(1)
        
        # Print some summary information if vorbose = True
        if(verbose):
            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")
        
        # Return the dataframe with missing value information
        return mis_val_table_ren_columns

## Function to remove columns with missing values greater than a threshold.
def remove_missing_value_columns(df, threshold):
    missing_values = compute_missing_values(df)
    missing_greater = missing_values[missing_values.iloc[:,1] >= threshold]
    df = df.drop(columns=missing_greater.index)
    print('{} columns have been dropped from input data set'.format(len(missing_greater)))
    del missing_values
    gc.collect()
    return df

####Functions used for plotting and visualizing outliers

## adds noise to y axis to avoid overlapping of data points
def rand_jitter(arr):
    return arr + np.random.randn(len(arr))

## plots distribution by target values and saves it in the 'CodeOutput' folder as png
def plot_feature_distribution(df, column):
    column_values = df[df[column].notna()][column]
    # group by target
    class_0_values = df[df[column].notna() & (df['TARGET']==0)][column]
    class_1_values = df[df[column].notna() & (df['TARGET']==1)][column]
    class_t_values = df[df[column].notna() & (df['TARGET'].isna())][column]        
    # for features with unique values >= 10
    if len(df[column].value_counts().keys()) >= 10:
        fig, ax = plt.subplots(1, figsize=(15, 4))
        if df[column].dtype == 'object':
            label_encoder = LabelEncoder()
            label_encoder.fit(column_values)
            class_0_values = label_encoder.transform(class_0_values)
            class_1_values = label_encoder.transform(class_1_values)
            class_t_values = label_encoder.transform(class_t_values)
            column_values = label_encoder.transform(column_values)
            plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, fontsize=12, rotation='vertical')

        ax.scatter(class_0_values, rand_jitter([0]*class_0_values.shape[0]), label='Class0', s=10, marker='o', color='#7ac143', alpha=1)
        ax.scatter(class_1_values, rand_jitter([10]*class_1_values.shape[0]), label='Class1', s=10, marker='o', color='#fd5c63', alpha=1)
        ax.scatter(class_t_values, rand_jitter([20]*class_t_values.shape[0]), label='Test', s=10, marker='o', color='#037ef3', alpha=0.4)
        ax.set_title(column +' group by target', fontsize=16)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(column +' distribution', fontsize=16)
    else:      
        all_categories = list(df[df[column].notna()][column].value_counts().keys())
        bar_width = 0.25
        
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.set_title(column, fontsize=16)
        plt.xlabel('Categories', fontsize=16)
        plt.ylabel('Counts', fontsize=16)

        value_counts = class_0_values.value_counts()
        x_0 = np.arange(len(all_categories))
        y_0 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_0, y_0, color='#7ac143', width=bar_width, label='class0')

        value_counts = class_1_values.value_counts()
        x_1 = np.arange(len(all_categories))
        y_1 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_1+bar_width, y_1, color='#fd5c63', width=bar_width, label='class1')
        
        value_counts = class_t_values.value_counts()
        x_2 = np.arange(len(all_categories))
        y_2 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_2+2*bar_width, y_2, color='#037ef3', width=bar_width, label='test')
        
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        
        for i, v in enumerate(y_0):
            if y_0[i]+y_1[i] == 0:
                ax.text(i - .08, max(y_0)//1.25,  'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i - .08, max(y_0)//1.25,  "{:0.1f}%".format(100*y_0[i]/(y_0[i]+y_1[i])), fontsize=14, rotation='vertical')
        
        for i, v in enumerate(y_1):
            if y_0[i]+y_1[i] == 0:
                ax.text(i - .08, max(y_0)//1.25,  'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i + bar_width - .08, max(y_0)//1.25, "{:0.1f}%".format(100*y_1[i]/(y_0[i]+y_1[i])), fontsize=14, rotation='vertical')
 
        for i, v in enumerate(y_2):
            if y_2[i] == 0:
                ax.text(i + 2*bar_width - .08, max(y_0)//1.25, 'Missing in Test', fontsize=14, rotation='vertical')
            else:
                ax.text(i + 2*bar_width - .08, max(y_0)//1.25, str(y_2[i]), fontsize=14, rotation='vertical')
        
        plt.xticks(x_0 + 2*bar_width/3, all_categories, fontsize=16)
        
    return plt


##Function to identify feature types in a given data frame : Categorical, floating point, integer and boolean
def identify_feature_types(df, features_to_ignore, verbose = True):
    categorical_features = list(f for f in df.select_dtypes(include='object') if f not in features_to_ignore)
    floatingPoint_features = list(f for f in df.select_dtypes(include='float64') if f not in features_to_ignore)
    temp = list(f for f in df.select_dtypes(include='int64') if f not in features_to_ignore)
    bool_features = [x for x in temp if 'FLAG' in x]
    integer_features = [x for x in temp if x not in bool_features]
    totalCount = len(categorical_features) + len(floatingPoint_features) + len(bool_features) + len(integer_features)
    if (verbose):
        print ('Catagorical Features : {}, Floating Point Features : {}, Boolean Features : {}, Integer Features : {}, Total Count : {}'
           .format(len(categorical_features), len(floatingPoint_features), len(bool_features), len(integer_features), totalCount))
    
    return categorical_features, floatingPoint_features, bool_features, integer_features

## Drop a given list of features(columns) from given data frame
def drop_features(df, features):
    df = df.drop(columns=[f for f in df.columns if f in features])
    return df

################################## Function used by data preperation part #######################



## Finction to scale numerical features in the data frame. Optianl list of features can be given
def scale_features(df, feature_list = None, scale_range = (0,1)):
    if(feature_list == None):
        feature_list = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'Unnamed :0']]
        
    #Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = scale_range)
    
    for feature in feature_list:
        if (df[feature].dtype == 'object'):
            continue

        scaler.fit(df[feature].values.reshape(-1,1))
        df[feature] = scaler.transform(df[feature].values.reshape(-1,1))
    
    return df

## Function to impute features in the date frame. Numerical features are imputed with median values and categorical with mean
## Optionally either numeric or categorical features can be imputed.
def impute_features(df, features = 'All'):
    categorical_feats, floatingPoint_feats, bool_feats, integer_feats = identify_feature_types(df,
                                                                                        ['TARGET', 'SK_ID_CURR', 'Unnamed :0'])
    if(features == 'All'):
        feature_list = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'Unnamed :0']]
    elif(features == 'Numerical'):
        feature_list = floatingPoint_feats + integer_feats
    elif(features == 'Categorical'):
        feature_list = categorical_feats + bool_feats
    else:
        raise ValueError('features can either be All, Numerical, Categorical')
        
    #Imputer for numerical features
    imputer = Imputer(strategy = 'median')
    
    for feature in feature_list:
        if (feature in categorical_feats + bool_feats):
            df[feature] = df[feature].fillna(df[feature].value_counts().index[0])
        else:
            imputer.fit(df[feature].values.reshape(-1,1))
            df[feature] = imputer.transform(df[feature].values.reshape(-1,1))

    return df

#Function to caluculate WOE for a feature. Nan values are considered.
def calculate_WOE(df, target,feature):
    lst = []
    for i in range(df[feature].nunique(dropna=False)):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (target == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (target == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    return data


#####Feature Selection Utils


# 1. Variance Selector
#    Note: 
#    - Scaling: no
#    - Impute missing values: yes
def var_selector(input_df, threshold = 0.01, verbose=True):
    df = input_df.copy()
    df = impute_features(df)
    selector = VarianceThreshold(threshold)
    selector.fit_transform(df)
    var_support = selector.get_support()
    var_feature = df.loc[:,var_support].columns.tolist()
    
    if(verbose):
        print('Variance : ', str(len(var_feature)), 'selected features out of', str(len(var_support)))
    
    del df, selector, var_feature
    return var_support

# 2. Pearson Correlation Selector
#    Note:
#    - Scaling: no
#    - Impute missing values: yes
def cor_selector(input_df, labels, drop_ratio = 0.5, verbose=True):
    drop_ratio = max(0,min(drop_ratio, 1))
    df = input_df.copy()
    cor_list = []
    df = impute_features(df, 'All')
    # calculate the correlation with y for each feature
    for i in df.columns.tolist():
        cor = np.corrcoef(df[i], labels)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = df.iloc[:,np.argsort(np.abs(cor_list))[-int(len(cor_list) * drop_ratio):]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in df.columns]
    if(verbose):
        print('Pearson : ', str(len(cor_feature)), 'selected features out of', str(len(cor_support)))
    
    del df, cor_feature
    return cor_support

# 3. Chi2 Selector
#    Note:
#    - Scaling: yes
#    - Impute missing values: yes
def chi_selector(input_df, labels, drop_ratio = 0.5, verbose=True):
    drop_ratio = max(0,min(drop_ratio, 1))
    df = input_df.copy()
    num_feats = int(len(df.columns) * drop_ratio)
    df = impute_features(df, features ='All')    
    df = scale_features(df)
    selector = SelectKBest(chi2, k=num_feats)
    selector.fit(df, labels)
    chi_support = selector.get_support()
    chi_feature = df.loc[:,chi_support].columns.tolist()
    
    if(verbose):
        print('CHI2 : ', str(len(chi_feature)), 'selected features out of', str(len(chi_support)))
    
    del df, selector, chi_feature
    return chi_support

# 4. RFE Selector
#    Note:
#    - Scaling: yes
#    - Impute missing values: yes 
def rfe_selector(input_df, labels, drop_ratio = 0.5, verbose=True):
    drop_ratio = max(0,min(drop_ratio, 1))
    df = input_df.copy()
    num_feats = int(len(df.columns) * drop_ratio)
    df = impute_features(df, features ='All')    
    df = scale_features(df)
    selector = RFE(estimator=LogisticRegression(solver='saga', n_jobs=-1), n_features_to_select=num_feats, step=10, verbose=20)
    selector.fit(df, labels)
    rfe_support = selector.get_support()
    rfe_feature = df.loc[:,rfe_support].columns.tolist()

    if(verbose):
        print('RFE : ', str(len(rfe_feature)), 'selected features out of', str(len(rfe_feature)))
    
    del df, selector, rfe_feature
    return rfe_support

# 5. Random Forest Selector
#    Note:
#    - Scaling: no
#    - Impute missing values: yes
def rf_selector(input_df, labels, drop_ratio = 0.5, verbose=True):
    drop_ratio = max(0,min(drop_ratio, 1))
    df = input_df.copy()
    df = impute_features(df, features ='All')
    selector = SelectFromModel(RandomForestClassifier(n_estimators=150), threshold='1.25*median')
    selector.fit(df, labels)
    rf_support = selector.get_support()
    rf_feature = df.loc[:,rf_support].columns.tolist()

    if(verbose):
        print('RF : ', str(len(rf_support)), 'selected features out of', str(len(rf_feature)))
    
    del df, selector, rf_feature
    return rf_support

# Function that runs feature 5 selection tests from above and returns a data frame of selected features
def select_features(input_df, labels, drop_ratio = 0.5):
    
    df = input_df.copy()
    
    var_support = var_selector(df)
    cor_support = cor_selector(df, labels, drop_ratio)
    chi_support = chi_selector(df, labels, drop_ratio)
    rfe_support = rfe_selector(df, labels, drop_ratio)
    rf_support = rf_selector(df, labels, drop_ratio)
    
    pd.set_option('display.max_rows', None)
    
    # put all selection together
    feature_selection_df = pd.DataFrame({'Feature':df.columns, 'Variance': var_support, 'Pearson':cor_support,
                                         'Chi-2':chi_support, 'RFE':rfe_support, 'Random Forest':rf_support})
    ## count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)

    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    
    del df, var_support, cor_support, chi_support, rfe_support, rf_support
    
    return feature_selection_df

# Function that samples original data set to either return balanced or stratified sample
def take_sample(df, labels, stratified = True):
    df['TARGET'] = labels
    num_samples = df.loc[df.TARGET==1].shape[0]
    if (stratified):
        sampling_ratio = (num_samples * 2) / df.shape[0]
        sample1 = df.loc[df.TARGET==1].sample(frac=sampling_ratio, replace=False)
        print('label 1 sample size:', str(sample1.shape[0]))
        sample0 = df.loc[df.TARGET==0].sample(frac=sampling_ratio, replace=False)
        print('label 0 sample size:', str(sample0.shape[0]))
    else:
        sample1 = df.loc[df.TARGET==1].sample(n=num_samples, replace=False)
        print('label 1 sample size:', str(sample1.shape[0]))
        sample0 = df.loc[df.TARGET==0].sample(n=num_samples, replace=False)
        print('label 0 sample size:', str(sample0.shape[0]))
    
    sampled_df = pd.concat([sample1, sample0], axis=0)
    sampled_labels = sampled_df.pop('TARGET')
    return sampled_df, sampled_labels