'''' 
Author: Ramine Tinati: raminetinati AT gmail dot com
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from pylab import rcParams

'''
A simple class to handle A simple Keras DNN Models Model
'''
class KerasModels:

    def __init__(self, learning_rate=0.01,
                 epochs=100,
                 batch_size=128,
                 display_step=100):

        print('Keras Models API Loading')
        # Training parameters.
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.input_dim = 0

    def keras_autoencoder_model(self, model_data):

        X_train, X_test, y_test = model_data
        X_train, X_test = np.array(X_train, np.float32), np.array(X_test, np.float32)

        self.input_dim = X_train.shape[1]

        encoding_dim = 14
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dense(encoding_dim, activation="tanh",
                        activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
        decoder = Dense(self.input_dim, activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.summary()
        return autoencoder

    def keras_metrics(self):

        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

        return METRICS

    def train_autoencoder_keras_model(self, autoencoder, model_data):

        metrics = self.keras_metrics()

        X_train, X_test, y_test = model_data
        X_train, X_test = np.array(X_train, np.float32), np.array(X_test, np.float32)

        nb_epoch = self.epochs
        batch_size = self.batch_size

        autoencoder.compile(optimizer='adam',
                            loss='mean_squared_error',
                            metrics=metrics)

        checkpointer = ModelCheckpoint(filepath="model.h5",
                                       verbose=0,
                                       save_best_only=True)

        tensorboard = TensorBoard(log_dir='./logs',
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)

        #kick off the training...
        history = autoencoder.fit(X_train, X_train,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  validation_data=(X_test, X_test),
                                  verbose=1,
                                  callbacks=[checkpointer, tensorboard]).history
        return autoencoder, history


    def keras_evaluation(self, history):

        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        metrics = ['loss', 'auc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history[metric], label='Train')
            plt.plot(history['val_' + metric], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])
            plt.legend()


    def auto_encoder_reconstruction_error(self, autoencoder, model_data):

        X_train, X_test, y_test = model_data

        predictions = autoencoder.predict(X_test)
        mse = np.mean(np.power(X_test - predictions, 2), axis=1)
        error_df = pd.DataFrame({'reconstruction_error': mse,
                                 'true_class': y_test})
        return error_df




    def reconstruction_error_plot(self, error_df, threshold):
        LABELS = ["Normal", "Fraud"]

        groups = error_df.groupby('true_class')
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="Fraud" if name == 1 else "Normal")
        ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.show()

        y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
        conf_matrix = confusion_matrix(error_df.true_class, y_pred)
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()


    def predict(self, autoencoder, data, threshold=4):

        # first obtain the features in a matrix and scaler
        if data.shape[1] != autoencoder.input.shape[1]:
            print('Input Dimension ({}) does not match network Input Dimension ({})'.format(data.shape[1],
                                                                                            self.input_dim))
            return data
        else:
            # data = StandardScaler().fit_transform(data)
            # Get the data ready for inferencing
            data_for_inference = data.values
            # Inference against the pretrained model
            predictions = autoencoder.predict(data_for_inference)
            # calculate the Mean Square Error against the overall set of data.
            mse = np.mean(np.power(data_for_inference - predictions, 2), axis=1)
            # now generate the predictions based ont he MSE and threshold
            y_pred = [1 if e > threshold else 0 for e in mse]
            # rejoin the preds back to the orginal data
            data['predictions'] = y_pred
            return data




            



