import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import numpy as np

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

def train_test_split_for_data(path: str, test_size: float):
    #Loading in the data
    data_df = pd.read_csv(path)
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

def train_test_split_for_data_autoencoder(path: str, test_size: float, semisupervised: bool):
    #Loading in the data
    data_df = pd.read_csv(path)
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

def train_test_split_count_matrix_baseline(path: str, test_size: float, semisupervised: bool=False):
    #Loading in the data
    data_df = pd.read_csv(path)
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
    y_test_success_np = y_train_success.to_list()
    y_test_fail_np = y_test_fail.to_list()
    y_test_np = y_test_success_np + y_test_fail_np
    if semisupervised:
        return x_train_success_np, x_test_np, y_test_np
    else:
        return x_train_np, x_test_np, y_test_np


