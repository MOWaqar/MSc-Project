"""
AUTHOR: M O WAQAR 20/08/2018

This script runs three classifiers 
1. Logistic Regression
2. SGD(LinearSVC)
3. LightGBM

They are run on the 17 datasats prepared with DataPreperation.py script.

****NOTE: If you have not executed 'DataPreperation.py' run that first and wait for it to finish before running this script.

********OUTPUT of this script is in the console in following format:

    LR Results:

Classifier could not run on the data set. Put X instead of score.
Classifier could not run on the data set. Put X instead of score.
Classifier could not run on the data set. Put X instead of score.
File: DataSetVersion2_a.csv,  AUC score : 0.607
File: DataSetVersion2_b.csv,  AUC score : 0.599
.
.
.
Logistic Regression - done in 499s

LinearSVC Results:

Classifier could not run on the data set. Put X instead of score.
Classifier could not run on the data set. Put X instead of score.
Classifier could not run on the data set. Put X instead of score.
File: DataSetVersion2_a.csv,  AUC score : 0.545
File: DataSetVersion2_b.csv,  AUC score : 0.534
.
.
.
Linear SVC - done in 12653s

LightGBM Results:

File: DataSetVersion1_a.csv,  AUC score : 0.750
File: DataSetVersion1_b.csv,  AUC score : 0.757
File: DataSetVersion1_c.csv,  AUC score : 0.758
.
.
.
"""

#Import Packages
import time
from contextlib import contextmanager
import gc
import UtilityFunctions as utils
import os

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

#Logistic regression
from sklearn.linear_model import LogisticRegression as LR

#SGD Classifier
from sklearn.linear_model import SGDClassifier

#LGBM Classifier
from lightgbm import LGBMClassifier

#ROC AUC performance metric
from sklearn.metrics import roc_auc_score
#Cross-Valdiation with stratified sampling
from sklearn.model_selection import StratifiedKFold

# Print time logs in console
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
#Read file from output data folder path. 
##Inputs: file name and folder path to location of the file
##Returns: data frame with features and labels column
def setup_input(filename, dataFolder):
    #Check if folder path is correct. If not throw an error stating so.
    if(not os.path.exists(dataFolder + os.sep + filename)):
        print("Input file not found. Make sure files and their path exist. Try running DataPreperation Script first.")
        raise NotADirectoryError
    
    # Read csv file, remove unecessary columns 'Unnamed: 0' and 'SK_ID_CURR' and extract and save labels column seperately
    df = pd.read_csv(dataFolder + os.sep + filename)
    labels = df.pop('TARGET')
    if('Unnamed: 0' in df):
        df = df.drop(columns='Unnamed: 0')
    if('SK_ID_CURR' in df):
        df = df.drop(columns='SK_ID_CURR')
        
    return df, labels


#Logistic Regression function with 10 fold stratified CV 
#Inputs: Data frame of features and labels column
#Outputs: Mean ROC-AUC score
def logistic_regression(df, labels):
    cv = StratifiedKFold(n_splits=10)
    foldScores = []

    for trainSet, testSet in cv.split(df, labels):
        clf = LR(C = 0.001)
        model = clf.fit(df.iloc[trainSet], labels.iloc[trainSet])
        probabilities = model.predict_proba(df.iloc[testSet])[:,1]
        foldScores.append(roc_auc_score(labels[testSet], probabilities))
        del clf
        
    return np.mean(foldScores)

#Linear SVC classifier with SGD with 10 fold stratified CV 
#Inputs: Data frame of features and labels column
#Outputs:ROC-AUC score
def svclassifier(df, labels):
    # Run classifier with 10 fold cross-validation
    cv = StratifiedKFold(n_splits=10)
    probas_ = np.zeros(df.shape[0])

    for trainSet, testSet in cv.split(df, labels):
        clf = SGDClassifier(alpha=5.5, class_weight='balanced', 
                            loss='hinge', max_iter=1000, n_jobs=-1)
        model = clf.fit(df.iloc[trainSet], labels.iloc[trainSet])
        probas_[testSet] = np.array(model.decision_function(df.iloc[testSet]))
        del clf
    
    return roc_auc_score(labels, probas_)

#Linear LGBM classifier with 10 fold stratified CV 
#Inputs: Data frame of features and labels column
#Outputs:ROC-AUC score
def LGBM_Classifier(df, labels):
    probas_ = np.zeros(df.shape[0])
    # Run classifier with cross-validation
    cv = StratifiedKFold(n_splits=10)

    for trainSet, testSet in cv.split(df, labels):
        clf = LGBMClassifier(n_jobs=-1, silent=True, )
        model = clf.fit(df.iloc[trainSet], labels.iloc[trainSet], eval_set=[(df.iloc[trainSet], labels.iloc[trainSet]),
            (df.iloc[testSet], labels.iloc[testSet])], eval_metric= 'auc', verbose= False, early_stopping_rounds= 200)
        probas_[testSet] = model.predict_proba(df.iloc[testSet], num_iteration=clf.best_iteration_)[:,1]
        del clf

    return roc_auc_score(labels, probas_)


##Main function executed when the script is called.
def main():
    #Initialize input and output folder paths
    input_folder, output_folder = utils.Initialize_Folder_Paths()
    
    #List of input files
    file_names = ['DataSetVersion1_a.csv', 'DataSetVersion1_b.csv', 'DataSetVersion1_c.csv', 'DataSetVersion2_a.csv',
              'DataSetVersion2_b.csv', 'DataSetVersion2_c.csv', 'DataSetVersion3_a.csv', 'DataSetVersion3_b.csv',
             'DataSetVersion3_c.csv', 'DataSetVersion4_a.csv', 'DataSetVersion4_b.csv', 'DataSetVersion4_c.csv',
             'DataSetVersion5_a.csv', 'DataSetVersion5_b.csv', 'DataSetVersion5_c.csv', 'DataSetVersion6_a.csv',
             'DataSetVersion6_b.csv']
    
    #Run LR with time log and print ROC-AUC scores for each file
    with timer("Logistic Regression"):
        print('\nLR Results:\n')
        for filename in file_names:
            input_df, labels = setup_input(filename, output_folder)
            try:
                print('File: {},  AUC score : {:0.3f}'.format(filename,logistic_regression(input_df, labels)))
            except:
                print('Classifier could not run on the data set. Put X instead of score.')
                continue
            finally:
                del input_df,labels
                gc.collect()
    
    #Run SGD(LinearSVC) with time log and print ROC-AUC scores for each file      
    with timer("Linear SVC"):
        print('\nLinearSVC Results:\n')
        for filename in file_names:
            input_df, labels = setup_input(filename, output_folder)
            try:
                print('File: {},  AUC score : {:0.3f}'.format(filename,svclassifier(input_df, labels)))
            except:
                print('Classifier could not run on the data set. Put X instead of score.')
                continue
            finally:
                del input_df,labels
                gc.collect()
                
    #Run LGBM with time log and print ROC-AUC scores for each file            
    with timer("LightGBM"):
        print('\nLightGBM Results:\n')
        for filename in file_names:
            input_df, labels = setup_input(filename, output_folder)
            try:
                print('File: {},  AUC score : {:0.3f}'.format(filename,LGBM_Classifier(input_df, labels)))
            except:
                print('Classifier could not run on the data set. Put X instead of score.')
                continue
            finally:
                del input_df,labels
                gc.collect()

if __name__ == "__main__":
    with timer("Calssification code"):
        main()