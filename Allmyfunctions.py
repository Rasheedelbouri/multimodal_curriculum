 #-*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:07:06 2017

@author: kebl4170
"""
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import sys
   
################################################################################################################################
    

def GetEncoded_Ward_Code(ward_codedf,alpha_uniq_trans_locs):
    Encoded_ward_codedf = np.zeros((len(ward_codedf),19))
    for i in range(0,len(ward_codedf)): # loop through every individual patients journey
        for k in range(0,ward_codedf.shape[1]-2):
            if ward_codedf[i:i+1][k][i] == 0: # skipping the line if a 0 is in the entry
                continue
        
            Encoded_ward_codedf[i:i+1][0][k] = alpha_uniq_trans_locs.loc[alpha_uniq_trans_locs.a == ward_codedf[i:i+1][k][i]].index[0] # Encoding the ward name by searching for its index value in alpha_uniq_trans_locs
            print(i)
            print(k) # monitoring progress
    return Encoded_ward_codedf

#####################################################################################################################################
    
def GetLogLikelihood(patient_journey, startprob, transition_matrix):
    
    Likelihood = np.zeros(len(patient_journey)) # initialise the vector
    log_likelihood = np.zeros(len(patient_journey))
    for i in range(0,len(patient_journey)):
        Init_prob = startprob[int(patient_journey[i:i+1][0][0])] # Extract the initial probability for the first location that appears
        if np.count_nonzero(patient_journey[i:i+1]) <= 1:
            Likelihood[i] = Init_prob   # setting likelihood as simply the initial probability if there is only one location in the journey
            
        else:
            Probs = np.zeros(np.count_nonzero(patient_journey[i:i+1]) - 1)
            for j in range(0,np.count_nonzero(patient_journey[i:i+1]) - 1):
                Probs[j] = transition_matrix[int(patient_journey[i:i+1][0][j]), int(patient_journey[i:i+1][0][j+1])] #extracting the probability of transition from the transition matrix
                
            Trans_probs = np.prod(Probs) # multiplying all of the transition probabilities
            Likelihood[i] = Init_prob*Trans_probs # multiplying the the transition probabilities by the starting location probability for overall likeliness of the journey
        
    
    return Likelihood
            
#####################################################################################################################################               

def GetHMMJourney(Encoded_ward_codedf, Duration_by_patient):
    
    journey = dict([]) # set up a dictionary for varying sized arrays to be generated
    for i in range(0,len(Encoded_ward_codedf)):
        
        n = np.count_nonzero(Encoded_ward_codedf[i:i+1]) # count how many locations are transferred to

        if n > 1: # if more than one location
            allArrays = np.array([])
            for j in range(0, int(n)):
                myArray = np.array([Encoded_ward_codedf[i:i+1][0][j]]*int(Duration_by_patient[i:i+1][j][i])) # generate the ward code repeated by how many hours are spent in the ward
                allArrays = np.concatenate([allArrays, myArray]) # concatenate these to make a single vector journey

            journey[i] = allArrays 

        elif n <= 1:
            for j in range(0,int(1)):
                
                journey[i] = np.array([Encoded_ward_codedf[i:i+1][0][j]]*int(Duration_by_patient[i:i+1][j][i])) # here we simply repeat the location by how many hours are spent there
        
        
    return journey
        
#####################################################################################################################################
def GetForwardProbs(Encoded_ward_codedf, Init_probs, transmat, alpha_uniq_trans_locs):
    
    n = int(np.count_nonzero(Encoded_ward_codedf))
    
    
    alpha = pd.DataFrame(np.zeros((len(alpha_uniq_trans_locs),n+1)))
    
    alpha[0] = Init_probs
    
    for i in range(1,int(n+1)):
        for j in range(0,len(alpha_uniq_trans_locs)):
            alpha[i][j] = alpha[i-1][j]*transmat[Encoded_ward_codedf[0][i-1], j]
            
            
        alpha[i] = alpha[i]/sum(alpha[i])
        
    return alpha    
#######################################################################################################################################
def GetForward(Encoded_ward_codedf, transmat, alpha_uniq_trans_locs):
    
    n = int(np.count_nonzero(Encoded_ward_codedf))
    
    
    alpha = pd.DataFrame(np.zeros((len(alpha_uniq_trans_locs),n+1)))
    interim = np.zeros(len(alpha_uniq_trans_locs))
    


    alpha[0] = np.zeros(len(alpha_uniq_trans_locs))
    alpha[0][Encoded_ward_codedf[0][0]] = 1
    
    for i in range(1,int(n+1)):
        for j in range(0,len(alpha_uniq_trans_locs)):


            interim = np.zeros(len(alpha_uniq_trans_locs))            
            for k in range(0,len(alpha_uniq_trans_locs)):
                
                interim[k] = alpha[i-1][k]*transmat[k,j]
                
            interim = sum(interim)
            
            alpha[i][j] = interim
            
        alpha[i] = alpha[i]/sum(alpha[i])
        
    return alpha

#######################################################################################################################################
    
def TestingForward(Encoded_ward_codedf, Forward_predicts):
    
    n = np.zeros(len(np.nonzero(Encoded_ward_codedf)[1]))
    
    
    for i in range(0,len(np.nonzero(Encoded_ward_codedf)[1])):
                
        if int(Encoded_ward_codedf[0][i]) in list(Forward_predicts[i].argsort()[-1:][::-1]):
            
            n[i]=1
            
            
    result = sum(n)/len(np.nonzero(Encoded_ward_codedf)[1])
    
    return result

#################################################################################################################

def GetDiagnosisJourney(diagnosisdf, transfersdf, subjectsdf, diagnosis_code, subject_history, entry_similarity):
    
    Condition_Diagnoses = diagnosisdf[diagnosisdf['diagnosis_code'].str.contains(diagnosis_code)]
    
    Admissions_due_to_condition = Condition_Diagnoses[Condition_Diagnoses['diagnosis_position'] == 1]
    
    Condition_transfers = transfersdf[transfersdf['hadm_id'].isin(Condition_Diagnoses['hadm_id'])]
    
    Condition_Admission_transfers = transfersdf[transfersdf['hadm_id'].isin(Admissions_due_to_condition['hadm_id'])]
    
    Condition_wardcode = Condition_transfers.groupby('hadm_id')['ward_code'].apply(lambda x: list(x)).reset_index()
    Condition_wardcode = pd.concat([Condition_wardcode, Condition_wardcode['ward_code'].apply(pd.Series).fillna(0)], axis=1)
    
    
    Cond_Adm_wardcode = Condition_Admission_transfers.groupby('hadm_id')['ward_code'].apply(lambda x: list(x)).reset_index()
    Cond_Adm_wardcode = pd.concat([Cond_Adm_wardcode, Cond_Adm_wardcode['ward_code'].apply(pd.Series).fillna(0)], axis=1)
    
    Condition_subjects = subjectsdf[subjectsdf['hadm_id'].isin(Condition_Diagnoses['hadm_id'])]
    Condition_subjects = Condition_subjects.groupby('subject_id')['hadm_id'].apply(lambda x: list(x)).reset_index()
    Condition_subjects = pd.concat([Condition_subjects, Condition_subjects['hadm_id'].apply(pd.Series).fillna(0)], axis=1) 
    
    Cond_Adm_subjects = subjectsdf[subjectsdf['hadm_id'].isin(Admissions_due_to_condition['hadm_id'])]
    Cond_Adm_subjects = Cond_Adm_subjects.groupby('subject_id')['hadm_id'].apply(lambda x: list(x)).reset_index()
    Cond_Adm_subjects = pd.concat([Cond_Adm_subjects, Cond_Adm_subjects['hadm_id'].apply(pd.Series).fillna(0)], axis=1) 

    
    
    for i in range(0,len(Cond_Adm_subjects)):
    
    
        n = int(np.count_nonzero(Cond_Adm_subjects[i:i+1])-2)
    
        submat = pd.DataFrame()

        for j in range(0,n):
            submat = submat.append(Cond_Adm_wardcode.loc[Cond_Adm_wardcode['hadm_id'] == Cond_Adm_subjects[i:i+1][j][i]], ignore_index=True)        
        
        print(i)    
        subject_history[i] = submat

    for i in range(0,len(subject_history)):
    
        subframe = pd.DataFrame()
        subframe['uniques'] = pd.Series(subject_history[i][0].unique())
        subframe['counts'] = subframe['uniques'].map(subject_history[i][0].value_counts()).fillna(0)
    
        entry_similarity[i] = subframe

    
    return subject_history, entry_similarity

#######################################################################################################################
    
def GetDiagnosisLoS(transfersdf, condition_ward_code, start_timedf, end_timedf):
        
    Condition_start_time = start_timedf[start_timedf['hadm_id'].isin(condition_ward_code['hadm_id'])]
    
    Condition_start_time = pd.concat([Condition_start_time, Condition_start_time['start_time'].apply(pd.Series).fillna(0)],axis=1)
    
    
    
    
    Condition_end_time = end_timedf[end_timedf['hadm_id'].isin(condition_ward_code['hadm_id'])]
    
    Condition_end_time = pd.concat([Condition_end_time, Condition_end_time['end_time'].apply(pd.Series).fillna(0)],axis=1)
   
    
    starttimedf = Condition_start_time.apply(pd.to_datetime, errors = 'coerce')
    tryout2 = starttimedf.apply(pd.to_datetime, errors = 'coerce')
    
    endtimedf = Condition_end_time.apply(pd.to_datetime, errors = 'coerce')
    tryout = endtimedf.apply(pd.to_datetime, errors = 'coerce')
    
    Duration_by_patient = tryout - tryout2
    
    return Duration_by_patient
    

##############################################################################################################################
    
def round_timedelta(td, period): # td = series of timedeltas, period input as seconds
    
    vector = td / np.timedelta64(1, 's')
    period_seconds = period
    half_period_seconds = vector/2
    remainder = vector % period_seconds

    #period_seconds = period.total_seconds()
    #half_period_seconds = period_seconds/2
    remainder = vector % period_seconds
    #if remainder >= half_period_seconds:
     #   return dt.timedelta(seconds=td.total_seconds() + (period_seconds -remainder))
    #else:
     #   return dt.timedelta(seconds = td.total_seconds() - remainder)
    
    vector[remainder >= half_period_seconds] = vector[remainder >= half_period_seconds] + (period_seconds - remainder[remainder >=half_period_seconds])
    vector[remainder<half_period_seconds] = vector[remainder<half_period_seconds] - remainder[remainder<half_period_seconds]
    
    
    
    return(vector)
    
##############################################################################################################################

def GetLoSDistribution(LoS_df, position_in_journey):
    
    LoS_distribution = pd.DataFrame(LoS_df[position_in_journey])
    
    unique_LoS = pd.DataFrame(LoS_distribution[position_in_journey].unique())
    
    sorted_LoS = unique_LoS.sort_values(0)
    
    sorted_LoS['count'] = sorted_LoS[0].map(LoS_df[position_in_journey].value_counts()).fillna(0)
    
    return sorted_LoS
    
##############################################################################################################################
def GetAUROC(model, test_x, binarized_test, test_y):
    
    
    n_classes = binarized_test.shape[1]
    y_score = pd.DataFrame(model.decision_function(test_x))
    
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y.iloc[:, i], y_score.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return(roc_auc)
    
################################################################################

def Get_Mahalanobis(dataframe):
    
    #dataframe = dataframe.loc[:, (dataframe!=0).any(axis=0)]
    dataframe = dataframe.reset_index(drop=True)
    cols = list(dataframe)
    nunique = dataframe.apply(pd.Series.nunique)
    if dataframe.shape[1] >= 15:
        cols_to_drop = nunique[nunique <= 2].index
        dataframe = dataframe.drop(cols_to_drop, axis=1)

    features = list(dataframe)
    means = pd.DataFrame(np.zeros(len(features)))
    covariance = np.cov(dataframe.T)
    inv_cov = np.linalg.inv(covariance)
    Mahalanobis = np.zeros(len(dataframe))
    
    

    for j in range(0,len(means)):
            means[0][j] = np.mean(dataframe.iloc[:,j])
            
    means = means.reset_index(drop=True)
    
    for i in range(0,len(dataframe)):
        first = pd.DataFrame(dataframe.iloc[i,:]).reset_index(drop=True)    
        
        V = first[i]-means[0]
        Mahalanobis[i] = np.sqrt(np.dot(np.dot(V.T,inv_cov), V))#[0][0]
        print(i)
        
    return(Mahalanobis, features)


def Get_reference_Mahalanobis(dataframe, reference):
    
    #dataframe = dataframe.loc[:, (dataframe!=0).any(axis=0)]
    
    reference = reference.reset_index(drop=True)
    cols = list(dataframe)
    nunique = reference.apply(pd.Series.nunique)
    if reference.shape[1] >= 15:
        cols_to_drop = nunique[nunique <= 2].index
        reference = reference.drop(cols_to_drop, axis=1)

    features = list(reference)
    means = pd.DataFrame(np.zeros(len(features)))
    covariance = np.cov(reference.T)
    inv_cov = np.linalg.inv(covariance)
    
    dataframe = dataframe.reset_index(drop=True)
    nunique = dataframe.apply(pd.Series.nunique)
    if dataframe.shape[1] >= 15:
        cols_to_drop = nunique[nunique <= 2].index
        dataframe = dataframe.drop(cols_to_drop, axis=1)

    features = list(dataframe)
    Mahalanobis = np.zeros(len(dataframe))
    
    

    for j in range(0,len(means)):
            means[0][j] = np.mean(reference.iloc[:,j])
            
    means = means.reset_index(drop=True)
    
    for i in range(0,len(dataframe)):
        first = pd.DataFrame(dataframe.iloc[i,:]).reset_index(drop=True)    
        
        V = first[i]-means[0]
        Mahalanobis[i] = np.sqrt(np.dot(np.dot(V.T,inv_cov), V))#[0][0]
        print(i)
        
    return(Mahalanobis, features)
###############################################################################

def Get_H_features(): # function which does the feature processing. Much quicker to make this into a function so memory is not used up
    #admissionsdf = pd.read_csv("admissions.csv",sep=';'); # importing data sets from HAVEN

    def clean_data(matrix): # function to eliminate duplication of columns as data crunching is carried out
        _, i = np.unique(matrix.columns, return_index=True)# finding unique column names
        matrix = matrix.iloc[:, i] # only re-writing one of each unique column name
            
        return(matrix)
    diagnosisdf = pd.read_csv("../../ORCHID_data/H/diagnosis.csv",sep=';');
    subjectsdf = pd.read_csv("../../ORCHID_data/H/subjects.csv",sep=';');
    consultantsdf = pd.read_csv('../../ORCHID_data/H/consultants.csv', sep=';')
        
    reduced_emergencydf = pd.read_csv('../../ORCHID_data/H/Reduced_emergencydf_for_svm.csv',sep=',').drop('Unnamed: 0',axis=1) #read the saved files of admissions corresponding to emergency patients
    
    
    secondary_locations = pd.read_csv('../../ORCHID_data/H/Secondary_locations_for_regression.csv',sep=',',header=None).drop(0,axis=1)# read the file of where they were transferred to after first ward
    
    reduced_subjectsdf = subjectsdf[subjectsdf['hadm_id'].isin(reduced_emergencydf['hadm_id'])].sort_values('hadm_id').reset_index().drop('index',axis=1); #finding the subjects who correspond to the same patients
    

    reduced_consultantsdf = consultantsdf[consultantsdf['hadm_id'].isin(reduced_emergencydf['hadm_id'])]
    reduced_consultantsdf = reduced_consultantsdf[reduced_consultantsdf['episode_number'] ==1]
    reduced_consultantsdf = reduced_consultantsdf.drop_duplicates('hadm_id')
    reduced_consultantsdf = reduced_consultantsdf.sort_values('hadm_id').reset_index(drop=True) # finding data from consultants df that goes into this patient subgroup
    
    
    reduced_consultantsdf['specialty_code'].replace('   ', np.nan, inplace=True)
    reduced_consultantsdf['specialty_code'] = pd.to_numeric(reduced_consultantsdf['specialty_code'],errors = 'coerce') # eliminating noisy data
    
    reduced_consultantsdf['treatment_function_code'].replace('   ', np.nan, inplace=True)
    reduced_consultantsdf['specialty_code'] = pd.to_numeric(reduced_consultantsdf['specialty_code'],errors = 'coerce') # eliminating noisy data

        
    Y = pd.concat([reduced_emergencydf, reduced_subjectsdf],axis=1); # concatenating dataframes to have it in one feature space
    
    Y['gender'] = Y['gender'].map({'F': 1, 'M': 0}) # encoding gender
    Y['ethnic_category'] = Y['ethnic_category'].map({'A ':0, 'B ':1, 'C ':2, 'D ':3, 'E ':4, 'F ':5, 'G ':6, 'H ':7,'J ':8, 'K ':9, 'L ':10, 'M ':11, 'N ':12, 'P ':13, 'R ':14, 'S ':15, 'Z ':16, '99':16}) # encoding ethnicity
    
    
    Y['secondary_location'] = secondary_locations[1];
    Y['specialty_code'] = pd.DataFrame(reduced_consultantsdf['specialty_code'])
    Y['treatment_function_code'] = reduced_consultantsdf['treatment_function_code'] # appending features to the feature space
    
    Y= clean_data(Y) # eliminating any duplication that may have occurred so far
    
    Hamza = pd.read_csv('../../ORCHID_data/H/JR_FEATURE_MX.csv',sep=';') # reading Hamza's features into the dataset
    ED_to_admission = pd.read_csv('../../ORCHID_data/H/ed2inpatient_hadmid.csv',sep=';')  # reading in the conversion between ED attendance and admission ID
    ED_to_admission = ED_to_admission[ED_to_admission['hadm_id_derived'] != 0 ] # choosing only entries we can use
    ED_to_admission = ED_to_admission.sort_values('edattendance_id').reset_index(drop=True) # sorting order so it is consistent with other matrix
    
    
    sub_Hamza = Hamza[Hamza['edattendance_id'].isin(ED_to_admission['edattendance_id'])].sort_values('edattendance_id').reset_index(drop=True)# finding the entries we have admission IDs for
    ED_to_admission  = ED_to_admission[ED_to_admission['edattendance_id'].isin(sub_Hamza['edattendance_id'])].reset_index(drop=True) #filtering the admission ID's we have
    
    if sum(ED_to_admission['edattendance_id'] - sub_Hamza['edattendance_id']) != 0: 
        sys.exit() # checking the dataframes are in the same order and if not exiting the program
    
    sub_Hamza.insert(0, 'hadm_id', ED_to_admission['hadm_id_derived']) #inserting admission IDs as the first column for comaprison to other features
    
    sub_Hamza = sub_Hamza[sub_Hamza['hadm_id'].isin(Y['hadm_id'])].reset_index(drop=True) # Extracting the features that we have from before
    
    
    sub_Y = Y[Y['hadm_id'].isin(sub_Hamza['hadm_id'])].reset_index(drop=True)   # extracting the datasets that we have admission IDs for
    sub_Hamza = sub_Hamza[sub_Hamza['hadm_id'].isin(sub_Y['hadm_id'])].reset_index(drop=True) # recursive filtering
    sub_Hamza = sub_Hamza.drop_duplicates('hadm_id').sort_values('hadm_id').reset_index(drop=True) #dropping the admission IDs that are repeating
    
    if sum(sub_Y['hadm_id'] - sub_Hamza['hadm_id']) != 0: 
        sys.exit() # another check on the dataframes to make sure the orders have not changed
        
    Y_with_Hamzas = pd.concat([sub_Y, sub_Hamza],axis=1); # concatenating the ordered matrices together
    
    Y = Y.dropna(axis=0, how='any') # dropping any rows which contain NAN data entries
    Y_with_Hamzas = Y_with_Hamzas.dropna(axis=0, how='any') # dropping any rows which contain NAN from the concatenated dataframe
    
    unique_values = Y_with_Hamzas['initial_location'].unique()
    unique_values = pd.DataFrame(unique_values)
    unique_values['count'] = unique_values[0].map(Y_with_Hamzas['initial_location'].value_counts())

    test_for_strat = unique_values[unique_values['count'] == 1].reset_index(drop=True)
    Y_with_Hamzas = Y_with_Hamzas[~Y_with_Hamzas['initial_location'].isin(list(test_for_strat[0]))]
    
    
    Y_with_Hamzas = clean_data(Y_with_Hamzas)
    
    Y_with_Hamzas = Y_with_Hamzas.reset_index(drop=True).sort_values('hadm_id')
    
    
    diagnosisdf = diagnosisdf[diagnosisdf['hadm_id'].isin(Y_with_Hamzas['hadm_id'])]
    
    patient_diagnoses = diagnosisdf.groupby('hadm_id')['diagnosis_code'].apply(lambda x: list(x)).reset_index()
    
    patient_diagnoses = patient_diagnoses.sort_values('hadm_id')
    
    if sum(Y_with_Hamzas['hadm_id'] - patient_diagnoses['hadm_id']) != 0: 
        sys.exit('Mismatch in the admission ID ordering: exit programme') # another check on the dataframes to make sure the orders have not changed
    
    diagnosis_count = np.zeros(len(patient_diagnoses))
    for i in range(0,len(patient_diagnoses)): 
        diagnosis_count[i] = len(set(patient_diagnoses['diagnosis_code'][i:i+1][i]))
    
    
    Y_with_Hamzas['diagnosis_count'] = diagnosis_count
    
    
    
    
    
    return(Y, Y_with_Hamzas)

###############################################################################
    
def Get_H_wardtypes(locations_vector, reduced = True):
    
    
    

    ward_types = ['Ward5', 'EAU', 'Cardiac', 'SEU', 'Womens', 'Endoscopy', 'ED'
                  , 'Immunology', 'Pregnancy', 'Mortuary', 'DaycaseSurgery', 'Rest'
                  , 'Geriatrics', 'Neuro', 'SpecialSurgery', 'MRI', 'Radiology'
                  , 'Discharge', 'TRAUMA', 'Ward6', 'Vasculars', 'ICU', 'Ward7', 'Paeds'
                  , 'Oncology', 'Ophthalmology','Renal']
            
    ward_indices = [[75,76,107,108,109,110,111,112,113,114,115],
                    [139,61,62],
                    [77,78,79,80,81,82,83,84,85,86,131,132,133],
                    [166,167,168,169],
                    [87,89,104,140,141,49],
                    [88,48],
                    [102,54],
                    [90],
                    [128,138,142,143,147,148,149,150,151,172,178,56,59,64,65,68],
                    [103],
                    [93,91,97,99,173,47,50,53,72,73],
                    [164,171,182,55,60,63,66,67,69,71],
                    [92,98,125,129,137,146],
                    [94,95,105,153,154,155,156,157,158,159],
                    [170],
                    [96],
                    [101],
                    [175],
                    [100,126,106,161,176,177,70,74],
                    [116,117,118,119],
                    [179,180,181],
                    [127,160,58],
                    [120,121,122,123,124],
                    [130,134,135,136,144,145,152,162,163,165,174,57],
                    [46],
                    [51],
                    [52]]
    for i in range(0,len(ward_types)):
        for j in range(0,len(ward_indices[i])):
                    
            locations_vector['initial_location']=np.where(locations_vector['initial_location'] ==ward_indices[i][j], i, locations_vector['initial_location'])
    
    if reduced == True:
        locations_vector['initial_location'][locations_vector['initial_location'].isin([4,5,6,8,10,11,12,14,17,19,23])] = 99
        
    return(locations_vector)
    

def readTable(tablename, separator):
    output = pd.read_csv(tablename, sep = separator)
    
    return(output)
    

       
def Get_ORCHID_wardtypes(locations_vector, reduced = True):
    ward_types = ['Ward5', 'EAU', 'Cardiac', 'SEU', 'Womens', 'Endoscopy', 'ED'
                  , 'Immunology', 'Pregnancy', 'Mortuary', 'DaycaseSurgery', 'Rest'
                  , 'Geriatrics', 'Neuro', 'SpecialSurgery', 'MRI', 'Radiology'
                  , 'Discharge', 'TRAUMA', 'Ward6', 'Vasculars', 'ICU', 'Ward7', 'Paeds']
            
    ward_indices = [[73,74,113,114,115,116,117,118,119,120,121],
                    [149,103],
                    [75,76,77,78,79,80,81,82,83,84,142,143,144],
                    [108,176,177,178,179],
                    [85,87,111,150,151],
                    [86],
                    [101],
                    [88,89],
                    [135,148,152,157,158,159,160,161,182,188,109,110,104],
                    [102],
                    [90,92,96,98,183],
                    [174,181,191],
                    [91,97,131,136,156],
                    [93,94,105,163,164,165,166,167,168,169],
                    [180,106],
                    [95],
                    [100],
                    [185],
                    [99,133,112,171,186,187,132],
                    [122,123,124],
                    [179,181],
                    [134,170],
                    [125,126,127,128,129,130,138,139,140,141],
                    [137,145,146,147,153,155,162,172,173,175,184,107]]
    
    for i in range(0,len(ward_types)):
        for j in range(0,len(ward_indices[i])):
                    
            locations_vector['initial_location']=np.where(locations_vector['initial_location'] ==ward_indices[i][j], i, locations_vector['initial_location'])
    
    if reduced == True:
        locations_vector['initial_location'][locations_vector['initial_location'].isin([4,5,6,8,10,11,12,14,17,19,23])] = 99
        
    return(locations_vector)
    

def readTable(tablename, separator):
    output = pd.read_csv(tablename, sep = separator)
    
    return(output)
    
def Standardise_data(dataframe): # function to standardis input data for SVM
    
    for i in range(0,len(dataframe.columns)):
        mean = np.mean(dataframe.iloc[:,i]) # finding the mean of every column
        sd = np.sqrt(np.var(dataframe.iloc[:,i])) # finding the standard deviation of every column
        
        if np.max(dataframe.iloc[:,i]) == 1:
            continue        
        
        elif sd == 0:
            dataframe.iloc[:,i] = dataframe.iloc[:,i] # leaving data as is if s.d = 0 
        else:
            dataframe.iloc[:,i] = (dataframe.iloc[:,i] - mean)/sd # centering data around 0
        print(i)
        
        
    return(dataframe)
