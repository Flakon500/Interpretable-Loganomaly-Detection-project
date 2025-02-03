import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import ast
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import Dense, Input
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from keras.models import Sequential


class AutoEncoder:
    def __init__(self, feature_dictionary: dict):
        self.feature_dictionary = feature_dictionary
        #Exception handling in case the dictionary does not have the correct keys or key lengths
        try:
            keys = ['number of neurons', 'activation function', 'kernel initializer', 'learning rate', 'loss function']
            for key in keys:
                if key not in feature_dictionary:
                    raise ValueError('Missing key in dictionary: ' + key)
            l = len(feature_dictionary[keys[0]])
            if len(feature_dictionary[keys[1]]) != l:
                raise ValueError('Different lengths in dictionary for keys: ' + keys[0]  + ' and ' + keys[1])
            if len(feature_dictionary[keys[2]]) != l:
                raise ValueError('Different lengths in dictionary for keys: ' + keys[1]  + ' and ' + keys[2])
        except ValueError as error:
            print(error.args)


        #Creating the model from the dictionary
        self.model = Sequential()
        #The first layer gets the input_shape parameter
        self.model.add(tf.keras.Input(shape=(feature_dictionary['number of neurons'][0],)))
        #The rest does not
        for i in range(0,len(feature_dictionary['number of neurons']),1):
            self.model.add(Dense(units = feature_dictionary['number of neurons'][i], 
                            activation = feature_dictionary['activation function'][i],  
                            kernel_initializer = feature_dictionary['kernel initializer'][i]))
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate = feature_dictionary['learning rate'])
        self.model.compile(loss = feature_dictionary['loss function'], optimizer=custom_optimizer)

    def fit(self, fit_feature_dict: dict):


        #Define the types of the inputs
        expected_types = {'x_train': pd.DataFrame,
                          'epochs': int,
                          'batch size': int,
                          'validation split': float,
                          'path': str
                          }
        #Chech whether the keys have the correct names and types
        try:
            for key, expected_type in expected_types.items():
                if key in fit_feature_dict:
                    if not isinstance(fit_feature_dict[key], expected_type):
                        raise TypeError(f"Key '{key}' is expected to be of type {expected_type.__name__}, "
                                        f"but got {type(fit_feature_dict[key]).__name__}.")
                else:
                    raise KeyError(f"Missing key: '{key}' in the input dictionary.")
        except (TypeError, KeyError) as error:
            print(f"Validation error: {error}")

        early_stopping = EarlyStopping(patience=10, verbose=1)
        checkpointer = ModelCheckpoint(filepath=fit_feature_dict['path'], save_best_only=True, verbose=1)




        history = self.model.fit(x=fit_feature_dict['x_train'], 
                                 y=fit_feature_dict['x_train'], 
                                 epochs=fit_feature_dict['epochs'], 
                                 validation_split=fit_feature_dict['validation split'],
                                 callbacks=[checkpointer, early_stopping])
        self.model = load_model(fit_feature_dict['path'])
        return history
    
    def evaluate(self, evaluate_feature_dict):

        #Define the expected types:
        expected_types = {'x_test': pd.DataFrame,
                          'y_test': pd.Series,
                          'difference calculating function': 'function',
                          'threshold of difference': float,
                          'with confusion matrix': bool}

        #Check whether the keys have the correct names and types
        try:
            for key, expected_type in expected_types.items():
                if key in evaluate_feature_dict:
                    if not isinstance(evaluate_feature_dict[key], expected_type):
                        raise TypeError(f"Key '{key}' is expected to be of type {expected_type.__name__}, "
                                        f"but got {type(evaluate_feature_dict[key]).__name__}.")
                else:
                    raise KeyError(f"Missing key: '{key}' in the input dictionary.")
        except (TypeError, KeyError) as error:
            print(f"Validation error: {error}")

        x_test = evaluate_feature_dict['x_test']
        y_test = evaluate_feature_dict['y_test']
        difference_calculating_function = evaluate_feature_dict['difference calculating function']
        threshold_of_difference = evaluate_feature_dict['threshold of difference']
        with_cm = evaluate_feature_dict['with confusion matrix']


        predictions = self.model.predict(x_test)
        real_count_vectors = x_test.to_numpy()
        difference_vector = difference_calculating_function(predictions, real_count_vectors)
        y_pred =  (difference_vector > threshold_of_difference).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        if with_cm:
            ConfusionMatrixDisplay(confusion_matrix=cm).plot()
            plt.title("Confusion Matrix")
            plt.show()
        precision = (cm[1,1]) / (cm[1,1] + cm[0,1])
        recall = (cm[1,1]) / (cm[1,1] + cm[1,0])
        f_score = 2*precision*recall / (precision+recall)
        accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
        true_positive_rate = cm[1,1] / (cm[1,0] + cm[1,1])
        false_positive_rate = cm[0,1] / (cm[0,1] + cm[0,0])

        return precision, recall, f_score, accuracy, true_positive_rate, false_positive_rate

    def plot_history(self, history, save_name:str):
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()
        plt.savefig(save_name)
    
    def create_f_score_recall_precision_plots(self, difference_calculating_function, x_test:pd.DataFrame, y_test:pd.Series, thresholds:np.ndarray, title:str, save_name:str):
        f_scores_list = []
        recalls_list = []
        precisions_list = []
        accuracy_list = []
        TPR = []
        FPR = []
        for threshold in thresholds:
            evaluate_feature_dict = {'x_test': x_test,
                                     'y_test': y_test,
                                     'difference calculating function': difference_calculating_function,
                                     'threshold of difference': threshold,
                                     'with confusion matrix': False}
            precision, recall, f_score, accuracy, true_positive_rate, false_positive_rate = self.evaluate(evaluate_feature_dict)
            f_scores_list.append(f_score)
            recalls_list.append(recall)
            precisions_list.append(precision)
            accuracy_list.append(accuracy)
            TPR.append(true_positive_rate)
            FPR.append(false_positive_rate)
        plt.plot(thresholds, f_scores_list, label = 'F-score')
        plt.plot(thresholds, precisions_list, label = 'Precision')
        plt.plot(thresholds, recalls_list, label = 'Recalls')
        plt.plot(thresholds, accuracy_list, label = 'Accuracy')
        plt.xlabel('Thresholds')
        plt.ylabel('Score value')
        plt.legend()
        plt.title(title)
        plt.savefig(save_name + 'Statistics.png')
        plt.show()
        plt.plot(FPR, TPR)
        plt.title('ROC-curve ' + title)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig(save_name + 'ROC_curve.png')

    def evaluate_count_matrix(self, x_test, y_test, threshold):
        predictions = np.zeros(len(x_test))
        for i in range(len(x_test)):
            errors = self.predict(x_test[i])
            if max(errors) > threshold:
                predictions[i] = 1
        cm = confusion_matrix(y_test, predictions)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.title("Confusion Matrix")
        plt.show()


        