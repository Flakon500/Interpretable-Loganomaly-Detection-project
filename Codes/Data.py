import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import numpy as np
from math import ceil
from typing import Tuple
from scipy.spatial.distance import cdist


def count_vector(event_list, n):
    vector = np.zeros(n)
    for event in event_list:
        index = int(event[1:]) - 1  # Convert 'E1' to 0, 'E2' to 1, etc.
        if 0 <= index < n:
            vector[index] += 1
    return vector

def one_hot_encoded_matrix(event_list, n):
    matrix = np.zeros([len(event_list), n])
    i = 0
    for event in event_list:
        index = int(event[1:]) - 1
        if 0 <= index < n:
            matrix[i,index] = 1
            i += 1
    return matrix

def count_matrix(event_list, n):
    matrix = np.zeros([len(event_list),n])
    i = 0
    for event in event_list:
        index = int(event[1:]) - 1
        if 0 <= index < n:
            if i > 0:
                matrix[i] = matrix[i-1]
            matrix[i, index] += 1
            i += 1
    return matrix

def train_test_split_for_data(path: str, test_size: float, BGL: bool=False):
    #Loading in the data
    data_df = pd.read_csv(path)
    if BGL:
        data_df.drop(columns='new label', inplace=True)
    #Converting the labels to binary numbers, 0 for success, 1 for failure
    mask = data_df['Final Label'] == 'Success'
    data_df.loc[mask, 'label'] = 0
    data_df.loc[~mask, 'label'] = 1
    #I do not need the index column
    data_df = data_df.reset_index(drop=True)

    #Once converted, the strings of the event IDs are also unnecessary
    data_df['Events']  = data_df['New Event ID'].apply(ast.literal_eval)
    #Calculating the maximum value of the En type events
    max_n = max(int(e[1:]) for sublist in data_df['Events'] for e in sublist)
    data_df['Event_Count_Vector'] = data_df['Events'].apply(lambda x: count_vector(x, max_n))
    data_df.drop(columns=['Final Label', 'Unnamed: 0', 'New Event ID', 'Events', 'BlockId'], inplace=True)
    count_vector_df = pd.DataFrame(data_df['Event_Count_Vector'].tolist(), index=data_df.index)

    #For logistic regression
    count_vector_df.columns = [f'feature_{i+1}' for i in range(count_vector_df.shape[1])]
    data_df = pd.concat([data_df, count_vector_df], axis=1)
    data_df.drop(columns='Event_Count_Vector', inplace = True)

    success_df = data_df.loc[data_df['label'] == 0].copy(deep=True)
    fail_df = data_df.loc[data_df['label'] == 1].copy(deep=True)
    success_df_label = success_df['label'].copy(deep=True)
    success_df.drop(columns='label', inplace=True)
    fail_df_label = fail_df['label'].copy(deep=True)
    fail_df.drop(columns='label', inplace=True)
    x_train_success, x_test_success,y_train_success, y_test_success = train_test_split(success_df, success_df_label, test_size=test_size, shuffle=True, random_state=42)
    x_train_fail, x_test_fail, y_train_fail, y_test_fail = train_test_split(fail_df, fail_df_label, test_size=test_size, shuffle = True, random_state=42)
    x_train = pd.concat([x_train_success, x_train_fail], ignore_index=True)
    x_test = pd.concat([x_test_success, x_test_fail], ignore_index=True)
    y_train = pd.concat([y_train_success, y_train_fail], ignore_index=True)
    y_test = pd.concat([y_test_success, y_test_fail], ignore_index=True)
    return x_train, x_test, y_train, y_test

def train_test_split_for_data_autoencoder(path: str, test_size: float, semisupervised: bool, BGL: bool=False):
    #Loading in the data
    data_df = pd.read_csv(path)
    if BGL:
        data_df.drop(columns='new label', inplace=True)
    #Converting the labels to binary numbers, 0 for success, 1 for failure
    mask = data_df['Final Label'] == 'Success'
    data_df.loc[mask, 'label'] = 0
    data_df.loc[~mask, 'label'] = 1
    #I do not need the index column
    data_df = data_df.reset_index(drop=True)

    #Once converted, the strings of the event IDs are also unnecessary
    data_df['Events']  = data_df['New Event ID'].apply(ast.literal_eval)
    #Calculating the maximum value of the En type events
    max_n = max(int(e[1:]) for sublist in data_df['Events'] for e in sublist)
    data_df['Event_Count_Vector'] = data_df['Events'].apply(lambda x: count_vector(x, max_n))
    data_df.drop(columns=['Final Label', 'Unnamed: 0', 'New Event ID', 'Events', 'BlockId'], inplace=True)
    count_vector_df = pd.DataFrame(data_df['Event_Count_Vector'].tolist(), index=data_df.index)

    #For logistic regression
    count_vector_df.columns = [f'feature_{i+1}' for i in range(count_vector_df.shape[1])]
    data_df = pd.concat([data_df, count_vector_df], axis=1)
    data_df.drop(columns='Event_Count_Vector', inplace = True)


    success_df = data_df.loc[data_df['label'] == 0].copy(deep=True)
    fail_df = data_df.loc[data_df['label'] == 1].copy(deep=True)
    success_df_label = success_df['label'].copy(deep=True)
    success_df.drop(columns='label', inplace=True)
    fail_df_label = fail_df['label'].copy(deep=True)
    fail_df.drop(columns='label', inplace=True)
    x_train_success, x_test_success,y_train_success, y_test_success = train_test_split(success_df, success_df_label, test_size=test_size, shuffle=True, random_state=42)
    x_train_fail, x_test_fail, y_train_fail, y_test_fail = train_test_split(fail_df, fail_df_label, test_size=test_size, shuffle = True, random_state=42)
    x_train = pd.concat([x_train_success, x_train_fail], ignore_index=True)
    x_test = pd.concat([x_test_success, x_test_fail], ignore_index=True)
    y_train = pd.concat([y_train_success, y_train_fail], ignore_index=True)
    y_test = pd.concat([y_test_success, y_test_fail], ignore_index=True)
    if semisupervised:
        return x_train_success, y_train_success, x_test_success, y_test_success, x_train_fail, y_train_fail, x_test_fail, y_test_fail
    else:
        return x_train, y_train, x_test, y_test

def train_test_split_count_matrix_baseline(path: str, test_size: float, semisupervised: bool=False, BGL: bool=False):
    #Loading in the data
    data_df = pd.read_csv(path)
    if BGL:
        data_df.drop(columns='new label', inplace = True)
    #Converting the labels to binary numbers, 0 for success, 1 for failure
    mask = data_df['Final Label'] == 'Success'
    data_df.loc[mask, 'label'] = 0
    data_df.loc[~mask, 'label'] = 1
    #I do not need the index column
    data_df = data_df.reset_index(drop=True)

    data_df['Events']  = data_df['New Event ID'].apply(ast.literal_eval)
    #Calculating the maximum value of the En type events
    max_n = max(int(e[1:]) for sublist in data_df['Events'] for e in sublist)
    data_df['Event_Count_Matrix'] = data_df['Events'].apply(lambda x: count_matrix(x, max_n))
    success_df = data_df.loc[data_df['label'] == 0].copy(deep=True)
    fail_df = data_df.loc[data_df['label'] == 1].copy(deep=True)
    success_df_label = success_df['label'].copy(deep=True)
    success_df.drop(columns='label', inplace=True)
    fail_df_label = fail_df['label'].copy(deep=True)
    fail_df.drop(columns='label', inplace=True)
    x_train_success, x_test_success, y_train_success, y_test_success = train_test_split(success_df, success_df_label, test_size=test_size, shuffle=True, random_state=42)
    x_train_fail, x_test_fail, y_train_fail, y_test_fail = train_test_split(fail_df, fail_df_label, test_size=test_size, shuffle = True, random_state=42)
    
    x_train_success_np = np.vstack(x_train_success['Event_Count_Matrix'].values)
    x_train_fail_np = np.vstack(x_train_fail['Event_Count_Matrix'].values)
    x_test_success_np = x_test_success['Event_Count_Matrix'].to_list()
    x_test_fail_np = x_test_fail['Event_Count_Matrix'].to_list()
    
    x_train_np = np.vstack((x_train_success_np, x_train_fail_np))
    x_test_np = x_test_success_np + x_test_fail_np
    
    y_test_success_np = y_test_success.to_list()
    y_test_fail_np = y_test_fail.to_list()
    y_test_np = y_test_success_np + y_test_fail_np
    if semisupervised:
        return x_train_success_np, x_test_np, y_test_np
    else:
        return x_train_np, x_test_np, y_test_np


def create_arrays_from_df(data_df: pd.DataFrame, length_of_matrices: int):
    X, y = [], []
    for _, row in data_df.iterrows():
        seq = row['Event_Count_Matrix']
        label = row['label']
            
        if seq.shape[0] < length_of_matrices:
            continue  # Discard short sequences
            
            # Create overlapping segments
        for i in range(ceil(seq.shape[0]/length_of_matrices)):
            if (i==0):
                X.append(seq[i*length_of_matrices : (i+1)*length_of_matrices])
            elif i<ceil(seq.shape[0]/length_of_matrices)-1:
                matrix_to_append = seq[i*length_of_matrices : (i+1)*length_of_matrices]
                vector_to_subtract = seq[i*length_of_matrices-1]
                X.append(matrix_to_append-vector_to_subtract)
            else:
                matrix_to_append = seq[-length_of_matrices:]
                vector_to_subtract = seq[-length_of_matrices-1]
                X.append(matrix_to_append-vector_to_subtract)

            y.append(label)
    return np.array(X), np.array(y)

#This function creates the arrays for training and testing purposes by padding the shorter matrices, and clipping the longer ones. In v1 the paddings are 0 vectors, in v2 the paddings are the last vectors repeated
def create_arrays_from_df_easier_v1(data_df: pd.DataFrame, length_of_matrices: int=50) -> Tuple[np.array, np.array]:
    X, y = [], []
    for _, row in data_df.iterrows():
        seq = row['Event_Count_Matrix']
        label = row['label']
        if seq.shape[0] < length_of_matrices:
            helper_matrix = np.zeros((length_of_matrices, seq.shape[1]))
            helper_matrix[:seq.shape[0]] = seq
        if seq.shape[0] >= length_of_matrices:
            helper_matrix = seq[:length_of_matrices]
        X.append(helper_matrix)
        y.append(label)
    return np.array(X), np.array(y)

def create_arrays_from_df_easier_v2(data_df: pd.DataFrame, length_of_matrices: int=50) -> Tuple[np.array, np.array]:
    X, y = [], []
    for _, row in data_df.iterrows():
        seq = row['Event_Count_Matrix']
        label = row['label']
        if seq.shape[0] < length_of_matrices:
            helper_matrix = np.zeros((length_of_matrices, seq.shape[1]))
            helper_matrix[:seq.shape[0]] = seq
            helper_matrix[seq.shape[0]:] = seq[-1]
        if seq.shape[0] >= length_of_matrices:
            helper_matrix = seq[:length_of_matrices]
        X.append(helper_matrix)
        y.append(label)
    return np.array(X), np.array(y)




def train_test_split_count_matrix_for_LSTM(path: str, length_of_matrices: int,sequence_col: str = 'Event_Count_Matrix', label_col: str='label', test_size: float=0.2, BGL: bool=False, random_state: int=42):
    #Loading in the data
    data_df = pd.read_csv(path)
    if BGL:
        data_df.drop(columns='new label', inplace = True)
    #Converting the labels to binary numbers, 0 for success, 1 for failure
    mask = data_df['Final Label'] == 'Success'
    data_df.loc[mask, 'label'] = 0
    data_df.loc[~mask, 'label'] = 1
    #I do not need the index column
    data_df = data_df.reset_index(drop=True)

    data_df['Events']  = data_df['New Event ID'].apply(ast.literal_eval)
    #Calculating the maximum value of the En type events
    max_n = max(int(e[1:]) for sublist in data_df['Events'] for e in sublist)
    data_df['Event_Count_Matrix'] = data_df['Events'].apply(lambda x: count_matrix(x, max_n))
    data_df.drop(columns=['Unnamed: 0', 'BlockId', 'New Event ID', 'Final Label', 'Events'], inplace=True)

    #Separate the dataframes based on the label and perform the train_test_split
    success_df = data_df.loc[data_df['label'] == 0].copy(deep=True)
    fail_df = data_df.loc[data_df['label'] == 1].copy(deep=True)
    train_success_df, test_success_df = train_test_split(success_df, test_size=test_size, shuffle=True, random_state=random_state)
    train_fail_df, test_fail_df = train_test_split(fail_df, test_size=test_size, shuffle=True, random_state=random_state)
    #Now concatenate the dataframes
    train_df = pd.concat([train_success_df, train_fail_df], ignore_index=True)
    test_df = pd.concat([test_success_df, test_fail_df], ignore_index=True)
    #Now let's perform the matrix creating magic
    X_train, y_train = create_arrays_from_df(train_df, length_of_matrices)
    X_test, y_test = create_arrays_from_df(test_df, length_of_matrices)
    
    return X_train, X_test, y_train, y_test


def count_matrix(event_list, n):
    matrix = np.zeros([len(event_list),n])
    i = 0
    for event in event_list:
        index = int(event[1:]) - 1
        if 0 <= index < n:
            if i > 0:
                matrix[i] = matrix[i-1]
            matrix[i, index] += 1
            i += 1
    return matrix


def train_test_split_count_matrix_for_LSTM_easier_padding(path: str, length_of_matrices: int, version: int=1,sequence_col: str = 'Event_Count_Matrix', label_col: str='label', test_size: float=0.2, BGL: bool=False, random_state: int=42):
    #Loading in the data
    data_df = pd.read_csv(path)
    if BGL:
        data_df.drop(columns='new label', inplace = True)
    
    #Converting the labels to binary numbers, 0 for success, 1 for failure
    mask = data_df['Final Label'] == 'Success'
    data_df.loc[mask, 'label'] = 0
    data_df.loc[~mask, 'label'] = 1
    #I do not need the index column
    data_df = data_df.reset_index(drop=True)

    data_df['Events']  = data_df['New Event ID'].apply(ast.literal_eval)
    #Calculating the maximum value of the En type events
    max_n = max(int(e[1:]) for sublist in data_df['Events'] for e in sublist)
    data_df['Event_Count_Matrix'] = data_df['Events'].apply(lambda x: count_matrix(x, max_n))
    data_df.drop(columns=['Unnamed: 0', 'BlockId', 'New Event ID', 'Final Label', 'Events'], inplace=True)

    #Separate the dataframes based on the label and perform the train_test_split
    success_df = data_df.loc[data_df['label'] == 0].copy(deep=True)
    fail_df = data_df.loc[data_df['label'] == 1].copy(deep=True)
    train_success_df, test_success_df = train_test_split(success_df, test_size=test_size, shuffle=True, random_state=random_state)
    train_fail_df, test_fail_df = train_test_split(fail_df, test_size=test_size, shuffle=True, random_state=random_state)
    #Now concatenate the dataframes
    train_df = pd.concat([train_success_df, train_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_success_df, test_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    #Now let's perform the matrix creating magic
    if version == 1:
        X_train, y_train = create_arrays_from_df_easier_v1(train_df, length_of_matrices)
        X_test, y_test = create_arrays_from_df_easier_v1(test_df, length_of_matrices)
    if version == 2:
        X_train, y_train = create_arrays_from_df_easier_v2(train_df, length_of_matrices)
        X_test, y_test = create_arrays_from_df_easier_v2(test_df, length_of_matrices)
    
    return X_train, X_test, y_train, y_test

####These are the functions that handle the sequences that are created for the embedding based models
def sequence_for_embedding(event_list, n, length_of_sequence: int=50) -> np.array:
    sequence = []
    i = 0
    for event in event_list:
        sequence.append(int(event[1:]) - 1)
    if len(sequence) < length_of_sequence:
        for i in range(length_of_sequence-len(sequence)):
            sequence.append(n)
        return np.array(sequence)
    else:
        return np.array(sequence[0:length_of_sequence])

def create_arrays_from_sequences(data_df:pd.DataFrame) -> Tuple[np.array, np.array]:
    X, y = [], []
    for _, row in data_df.iterrows():
        seq = row['Event_sequences']
        label = row['label']
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)


def create_sequences_for_embedding(path: str, length_of_sequences: int, test_size: float=0.2, BGL: bool=False, random_state: int=42):
    #Loading in the data
    data_df = pd.read_csv(path)
    if BGL:
        data_df.drop(columns='new label', inplace = True)
    #Converting the labels to binary numbers, 0 for success, 1 for failure
    mask = data_df['Final Label'] == 'Success'
    data_df.loc[mask, 'label'] = 0
    data_df.loc[~mask, 'label'] = 1
    #I do not need the index column
    data_df = data_df.reset_index(drop=True)

    data_df['Events']  = data_df['New Event ID'].apply(ast.literal_eval)
    #Calculating the maximum value of the En type events
    max_n = max(int(e[1:]) for sublist in data_df['Events'] for e in sublist)
    data_df['Event_sequences'] = data_df['Events'].apply(lambda x: sequence_for_embedding(x, max_n, length_of_sequences))
    data_df.drop(columns=['Unnamed: 0', 'BlockId', 'New Event ID', 'Final Label', 'Events'], inplace=True)
    #Separate the dataframes based on the label and perform the train_test_split
    success_df = data_df.loc[data_df['label'] == 0].copy(deep=True)
    fail_df = data_df.loc[data_df['label'] == 1].copy(deep=True)
    train_success_df, test_success_df = train_test_split(success_df, test_size=test_size, shuffle=True, random_state=random_state)
    train_fail_df, test_fail_df = train_test_split(fail_df, test_size=test_size, shuffle=True, random_state=random_state)
    #Now concatenate the dataframes
    train_df = pd.concat([train_success_df, train_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_success_df, test_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_train, y_train = create_arrays_from_sequences(train_df)
    X_test, y_test = create_arrays_from_sequences(test_df)
    return X_train, X_test, y_train, y_test