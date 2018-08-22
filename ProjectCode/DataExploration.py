"""
AUTHOR: M O WAQAR 20/08/2018

This script executed exploratory data analysis(EDA) and prepares data version for the project
The output of this script is saved in the Output Folder along with some information printed out in console.



"""

#Import packages
import UtilityFunctions as utils
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import gc


def check_data_stats(input_df):
    print(('Input file contains {} feature and {} data instances.\n').format(input_df.shape[1], input_df.shape[0]))
    print('Input is in the following form: \n')
    print(input_df.head())

    print('\nColumn types: \n')
    print(input_df.dtypes.value_counts())


def check_data_imbalance(input_df, save_to_folder=None):
    print('Data Imbalance:')

    value_df = input_df['TARGET'].value_counts().to_frame().reset_index()
    value_df.columns = ['Value', 'Count']
    value_df['% of Total Values'] = [100 * value_df['Count'][0] / input_df.shape[0], 100 * value_df['Count'][1] / input_df.shape[0]]
    print(value_df.head())

    #Generate pie plot and save it in outputFolder
    figure = value_df.plot.pie(y='% of Total Values', figsize= (5,5), explode=(0,0.1), autopct='%.1f%%').get_figure()

    if(save_to_folder != None):
        figure.savefig(save_to_folder + os.sep + 'DataImbalance.png', dpi=200, format='png')
    
    figure.show()
    
    del value_df



def check_missing_values(input_df, save_to_folder = None):
    # Missing values statistics
    missing_values = utils.compute_missing_values(input_df)
    
    if(save_to_folder != None):
        missing_values.head(10).to_csv(save_to_folder + os.sep + 'Top10MissingValues.csv')
        missing_values.tail(10).to_csv(save_to_folder + os.sep + 'Bottom10MissingValues.csv')

    print('Top 10 missing value columns: \n')
    print(missing_values.head(10))

    print('\nBottom 10 missing value columns: \n')
    print(missing_values.tail(10))

    del missing_values

def check_outliers(trian_df, test_df, save_to_folder=None):
    #join train and test data sets for better visualization of outliers
    combined_df = trian_df.append(test_df)
    print('\nJoined train rows {} with test rows {}. New total {}\n'.format(trian_df.shape[0], test_df.shape[0],combined_df.shape[0]))

    print('\nBefore replacing outliers.\n')
    
    plot1 = utils.plot_feature_distribution(combined_df, 'DAYS_EMPLOYED')
    
    if(save_to_folder != None):
        plot1.savefig(save_to_folder + '\\DAYS_EMPLOYED_distribution.png', bbox_inches='tight')
    
    plot1.show()

    plot2 = utils.plot_feature_distribution(combined_df, 'AMT_INCOME_TOTAL')
    
    if(save_to_folder != None):
        plot2.savefig(save_to_folder + '\\AMT_INCOME_TOTAL_distribution.png', bbox_inches='tight')
    
    plot2.show()

    plot1.close()
    plot2.close()

    combined_df = utils.replace_outliers(combined_df)


    print('\nAfter replacing outliers.\n')
    plot1 = utils.plot_feature_distribution(combined_df, 'DAYS_EMPLOYED')
    if(save_to_folder != None):
        plot1.savefig(save_to_folder + '\\DAYS_EMPLOYED_corrected_distribution.png', bbox_inches='tight')
   
    plot1.show()

    plot2 = utils.plot_feature_distribution(combined_df, 'AMT_INCOME_TOTAL')
    
    if(save_to_folder != None):
        plot2.savefig(save_to_folder + '\\AMT_INCOME_TOTAL_corrected_distribution.png', bbox_inches='tight')
    
    plot2.show()

    plot1.close()
    plot2.close()

    del combined_df



def check_feature_correlation(input_df, feature_Name = 'TARGET', save_to_folder = None):
    
    if(feature_Name not in input_df):
        print('\nCorrelation target feature not in data frame.\n')
        return

    input_df = utils.replace_outliers(input_df)
    # Find correlations with the target and sort
    correlations = input_df.corr()[feature_Name].sort_values()
    correlations = correlations.drop(index=feature_Name)

    # Display correlations
    print('\nMost Positive Correlations:\n', correlations.tail(15))
    print('\nMost Negative Correlations:\n', correlations.head(15))

    if(save_to_folder != None):
        strongest20 = pd.concat([correlations.head(10), correlations.tail(10)] , axis=0)
        strongest20.to_csv(save_to_folder + '\\Top20Correlations.csv')

    del strongest20

def check_KDE_plot(input_df, feature_Name='DAYS_BIRTH', save_to_folder=None):
    
    figure = plt.figure(figsize = (10, 8))
    
    # Days birth is divided by 365 to be converted to years.
    if(feature_Name == 'DAYS_BIRTH'):
        division_factor = 365
    else:
        division_factor = 1

    # KDE plot of loans that were repaid on time
    sns.kdeplot(input_df.loc[input_df['TARGET'] == 0, feature_Name] / division_factor, label = 'target == 0')

    # KDE plot of loans which were not repaid on time
    sns.kdeplot(input_df.loc[input_df['TARGET'] == 1, feature_Name] / division_factor, label = 'target == 1')

    # Labeling of plot
    plt.xlabel(feature_Name); plt.ylabel('Density'); plt.title('KDE plot of {}'.format(feature_Name));


    if(save_to_folder!=None):
        figure.savefig(save_to_folder + os.sep + feature_Name + '_plot-KDE.png', dpi=200, format='png')
        
    figure.show()



def run_full_eda(input_folder, output_folder=None):
    input_df = utils.read_csv_File(input_folder + '\\application_train.csv')
    test_df = utils.read_csv_File(input_folder + '\\application_test.csv')
    
    check_data_stats(input_df)
    check_data_imbalance(input_df, save_to_folder=output_folder)
    check_missing_values(input_df, output_folder)
    check_outliers(input_df, test_df, output_folder)

    input_df = utils.replace_outliers(input_df)
    
    check_feature_correlation(input_df, feature_Name = 'TARGET', save_to_folder = output_folder)
    check_KDE_plot(input_df, feature_Name='DAYS_BIRTH', save_to_folder=output_folder)
    check_KDE_plot(input_df, feature_Name='EXT_SOURCE_3', save_to_folder=output_folder)
    
    del input_df, test_df
    gc.collect()



def main():
    input_folder, output_folder = utils.Initialize_Folder_Paths()
    
    run_full_eda(input_folder, output_folder)


if __name__ == "__main__":
  main()


