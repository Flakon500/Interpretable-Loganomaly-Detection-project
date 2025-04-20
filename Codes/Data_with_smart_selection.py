import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import numpy as np
from math import ceil
from typing import Tuple
from scipy.spatial.distance import cdist

#Necessary function for creating count vectors
def count_vector(event_list, n):
    vector = np.zeros(n)
    for event in event_list:
        index = int(event[1:]) - 1  # Convert 'E1' to 0, 'E2' to 1, etc.
        if 0 <= index < n:
            vector[index] += 1
    return vector

#Function that creates one-hot encoded matrices
def one_hot_encoded_matrix(event_list, n):
    matrix = np.zeros([len(event_list), n])
    i = 0
    for event in event_list:
        index = int(event[1:]) - 1
        if 0 <= index < n:
            matrix[i,index] = 1
            i += 1
    return matrix

#Function that creates count matrices
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

#Function that creates arrays from dataframes
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

#Other function that creates arrays from dataframes
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

#And another one
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

def benchmark_selection(x_train_success_df: pd.DataFrame, number_to_choose: int):
    sampled_df = x_train_success_df.sample(n=int(len(x_train_success_df)*1/(1+number_to_choose)), replace=False, random_state=None).reset_index(drop=True)
    return sampled_df
#Function that creates the sequences for embedding
def create_sequences_for_embedding_less_data_benchmark(path: str, length_of_sequences: int, test_size: float=0.2, BGL: bool=False,number_to_choose: int=9,  random_state: int=42):
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
    train_success_df2 = benchmark_selection(train_success_df, number_to_choose)
    train_fail_df, test_fail_df = train_test_split(fail_df, test_size=test_size, shuffle=True, random_state=random_state)
    #Now concatenate the dataframes
    train_df = pd.concat([train_success_df2, train_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_success_df, test_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_train, y_train = create_arrays_from_sequences(train_df)
    X_test, y_test = create_arrays_from_sequences(test_df)
    return X_train, X_test, y_train, y_test

#Function that creates count matrices for LSTM

def train_test_split_count_matrix_for_LSTM_easier_padding_less_data_benchmark(path: str, length_of_matrices: int, version: int=1,sequence_col: str = 'Event_Count_Matrix', label_col: str='label', test_size: float=0.2, BGL: bool=False, number_to_choose: int=9, random_state: int=42):
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
    train_success_df2 = benchmark_selection(train_success_df, number_to_choose)

    train_fail_df, test_fail_df = train_test_split(fail_df, test_size=test_size, shuffle=True, random_state=random_state)
    #Now concatenate the dataframes
    train_df = pd.concat([train_success_df2, train_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_success_df, test_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    #Now let's perform the matrix creating magic
    if version == 1:
        X_train, y_train = create_arrays_from_df_easier_v1(train_df, length_of_matrices)
        X_test, y_test = create_arrays_from_df_easier_v1(test_df, length_of_matrices)
    if version == 2:
        X_train, y_train = create_arrays_from_df_easier_v2(train_df, length_of_matrices)
        X_test, y_test = create_arrays_from_df_easier_v2(test_df, length_of_matrices)
    
    return X_train, X_test, y_train, y_test


def train_test_split_count_matrix_for_LSTM_easier_padding_with_less_data_smart(path: str, length_of_matrices: int, version: int=1,sequence_col: str = 'Event_Count_Matrix', label_col: str='label', test_size: float=0.2,number_to_choose:int=9, BGL: bool=False, random_state: int=42):
    #Loading in the data
    data_df = pd.read_csv(path)
    
    #Converting the labels to binary numbers, 0 for success, 1 for failure
    mask = data_df['Final Label'] == 'Success'
    data_df.loc[mask, 'label'] = 0
    data_df.loc[~mask, 'label'] = 1
    #I do not need the index column
    data_df = data_df.reset_index(drop=True)

    data_df = data_df.reset_index(drop=True)
    data_df['Events']  = data_df['New Event ID'].apply(ast.literal_eval)

    success_df = data_df.loc[data_df['label'] == 0].copy(deep=True)
    fail_df = data_df.loc[data_df['label'] == 1].copy(deep=True)
    success_df = smart_selection(success_df, fail_df)
    train_success_df, test_success_df = train_test_split(success_df, test_size=test_size, shuffle=True, random_state=random_state)
    train_fail_df, test_fail_df = train_test_split(fail_df, test_size=test_size, shuffle=True, random_state=random_state)
    #Now concatenate the dataframes
    train_df = pd.concat([train_success_df, train_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_success_df, test_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    max_n = max(int(e[1:]) for sublist in data_df['Events'] for e in sublist)
    train_df['Event_Count_Matrix'] = train_df['Events'].apply(lambda x: count_matrix(x, max_n))
    train_df.drop(columns=['Unnamed: 0', 'BlockId', 'New Event ID', 'Final Label', 'Events'], inplace=True)

    test_df['Event_Count_Matrix'] = test_df['Events'].apply(lambda x: count_matrix(x, max_n))
    test_df.drop(columns=['Unnamed: 0', 'BlockId', 'New Event ID', 'Final Label', 'Events'], inplace=True)


    #Now let's perform the matrix creating magic
    if version == 1:
        X_train, y_train = create_arrays_from_df_easier_v1(train_df, length_of_matrices)
        X_test, y_test = create_arrays_from_df_easier_v1(test_df, length_of_matrices)
    if version == 2:
        X_train, y_train = create_arrays_from_df_easier_v2(train_df, length_of_matrices)
        X_test, y_test = create_arrays_from_df_easier_v2(test_df, length_of_matrices)
    
    return X_train, X_test, y_train, y_test

def select_based_on_anomalies(working_success_df: pd.DataFrame, working_fail_df: pd.DataFrame, number_to_remove: int) -> None:
    """The first part of the smart selection, it performs the following algorithm: Choose one anomalous datapoint, 
    Select the closest non-anomalous point to it
    Remove the closest number_to_remove many datapoints within the non-anomalous point's neighbourhood
    """    
    working_success_df_vectors = np.vstack(working_success_df['Event_Count_Vector'].values)  # shape: (n1, d)
    for _, row in working_fail_df.iterrows():
        vec = row['Event_Count_Vector'].reshape(1, -1)  # shape: (1, d)

        # Step 1: Compute cosine distances in bulk
        distances = cdist(working_success_df_vectors, vec, metric='cosine').flatten()

        # Mask out already marked vectors if necessary
        idx_min = distances.argmin()
        working_success_df.loc[idx_min, 'keep_it'] = 1

        # Step 2: Find number_to_remove closest with keep_it == 0
        mask = working_success_df['keep_it'] == 0
        available_indices = working_success_df.index[mask]

        if len(available_indices) > 0:
            # Compute distances only for unmarked vectors
            unmarked_vectors = working_success_df_vectors[mask.values]
            target_vec = working_success_df_vectors[idx_min].reshape(1, -1)

            distances_to_target = cdist(unmarked_vectors, target_vec, metric='cosine').flatten()

            k = min(number_to_remove, len(distances_to_target))  # handle corner case
            idxs_to_remove = available_indices[distances_to_target.argsort()[:k]]

            working_success_df.loc[idxs_to_remove, 'keep_it'] = -1

def select_remainder(working_success_df: pd.DataFrame, number_to_remove: int):
    working_success_df_vectors = np.vstack(working_success_df['Event_Count_Vector'].values)  # shape: (n1, d)

    while True:
        mask_remaining = working_success_df['keep_it'] == 0
        remaining_indices = working_success_df.index[mask_remaining]         # pandas indices
        remaining_positions = np.where(mask_remaining.values)[0]             # numpy positions (0..n-1)

        if len(remaining_indices) == 0:
            break  # All done!

        if len(remaining_indices) <= number_to_remove + 1:
            # Not enough left to fully remove neighbors
            chosen_pos = np.random.choice(remaining_positions)  # pick by position
            chosen_idx = working_success_df.index[chosen_pos]   # map back to pandas index

            working_success_df.loc[chosen_idx, 'keep_it'] = 1
            other_idxs = remaining_indices.drop(chosen_idx)
            working_success_df.loc[other_idxs, 'keep_it'] = -1
            break  # Finished after this

        # Step 1: Pick one randomly from remaining
        chosen_pos = np.random.choice(remaining_positions)
        chosen_idx = working_success_df.index[chosen_pos]
        working_success_df.loc[chosen_idx, 'keep_it'] = 1

        # Step 2: Compute distances from this vector to all remaining
        chosen_vec = working_success_df_vectors[chosen_pos].reshape(1, -1)
        unmarked_vectors = working_success_df_vectors[remaining_positions]

        distances = cdist(unmarked_vectors, chosen_vec, metric='cosine').flatten()

        # Step 3: Remove number_to_remove closest (excluding chosen itself)
        # Get positions of remaining without chosen
        positions_without_chosen = remaining_positions[remaining_positions != chosen_pos]

        idxs_to_remove_pos = positions_without_chosen[
            distances[remaining_positions != chosen_pos].argsort()[:number_to_remove]
        ]

        idxs_to_remove = working_success_df.index[idxs_to_remove_pos]  # map positions back to indices

        working_success_df.loc[idxs_to_remove, 'keep_it'] = -1


def smart_selection(x_train_success: pd.DataFrame, x_train_fail: pd.DataFrame, lengths: list = [13,25,35], number_to_remove: int=9):
    '''This function is the implemented version of the potentially smarter selection to use less datapoints in training
    x_train_success: The dataframe that only has non-anomaly type of data in it
    x_train_fail: The dataframe that only has anomalous data in it
    lengths: A list that consists of the lengths I want to divide my data into, an idea taken from stratified sampling
    number_to_remove: An integer number, that describes how many datapoint I remove from the dataset after choosing one
    '''
    #Performing the stratified sampling inspired step
    max_n = max(int(e[1:]) for sublist in x_train_success['Events'] for e in sublist)
    x_train_success['Event_Count_Vector'] = x_train_success['Events'].apply(lambda x: count_vector(x, max_n))
    x_train_fail['Event_Count_Vector'] = x_train_fail['Events'].apply(lambda x: count_vector(x, max_n))
    x_train_success['lengths'] = x_train_success['Event_Count_Vector'].apply(lambda x: sum(x))
    x_train_fail['lengths'] = x_train_fail['Event_Count_Vector'].apply(lambda x: sum(x))
    
    x_train_fail2 =x_train_fail.copy(deep = True)
    result_df = pd.DataFrame(columns=['Event_Count_Vector', 'lengths', 'keep_it'])

    for length in lengths:
        mask = x_train_success['lengths'] <= length
        working_success_df = x_train_success[mask].copy(deep=True)
        working_success_df = working_success_df.reset_index(drop=True)
        x_train_success = x_train_success[~mask]

        mask = x_train_fail2['lengths'] <= length
        working_fail_df = x_train_fail2[mask].copy(deep=True)
        x_train_fail2 = x_train_fail2[~mask]

        working_success_df['keep_it'] = 0

        select_based_on_anomalies(working_success_df, working_fail_df, number_to_remove)
        select_remainder(working_success_df, number_to_remove)

        result_df = pd.concat([result_df, working_success_df], ignore_index = True)
        print('lenght done: ', length)
    result_df.drop(columns='Event_Count_Vector', inplace=True)
    x_train_fail.drop(columns = 'Event_Count_Vector', inplace = True)
    mask = result_df['keep_it'] == 1
    result_df = result_df[mask]
    return result_df


def create_sequences_for_embedding_with_less_train_data_smart(path: str, length_of_sequences: int, number_to_choose: int=9, lengths: list=[13,25,35],  test_size: float=0.2, random_state: int=42):
    #Loading in the data
    data_df = pd.read_csv(path)
    #Converting the labels to binary numbers, 0 for success, 1 for failure
    mask = data_df['Final Label'] == 'Success'
    data_df.loc[mask, 'label'] = 0
    data_df.loc[~mask, 'label'] = 1
    #I do not need the index column
    data_df = data_df.reset_index(drop=True)
    data_df['Events']  = data_df['New Event ID'].apply(ast.literal_eval)

    success_df = data_df.loc[data_df['label'] == 0].copy(deep=True)
    fail_df = data_df.loc[data_df['label'] == 1].copy(deep=True)
    success_df = smart_selection(success_df, fail_df)
    train_success_df, test_success_df = train_test_split(success_df, test_size=test_size, shuffle=True, random_state=random_state)
    train_fail_df, test_fail_df = train_test_split(fail_df, test_size=test_size, shuffle=True, random_state=random_state)
    #Now concatenate the dataframes
    train_df = pd.concat([train_success_df, train_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_success_df, test_fail_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    



    #Calculating the maximum value of the En type events
    max_n = max(int(e[1:]) for sublist in data_df['Events'] for e in sublist)
    train_df['Event_sequences'] = train_df['Events'].apply(lambda x: sequence_for_embedding(x, max_n, length_of_sequences))
    train_df.drop(columns=['Unnamed: 0', 'BlockId', 'New Event ID', 'Final Label', 'Events'], inplace=True)

    test_df['Event_sequences'] = test_df['Events'].apply(lambda x: sequence_for_embedding(x, max_n, length_of_sequences))
    test_df.drop(columns=['Unnamed: 0', 'BlockId', 'New Event ID', 'Final Label', 'Events'], inplace=True)
    #Separate the dataframes based on the label and perform the train_test_split
    X_train, y_train = create_arrays_from_sequences(train_df)
    X_test, y_test = create_arrays_from_sequences(test_df)
    return X_train, X_test, y_train, y_test
