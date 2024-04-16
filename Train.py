import pandas as pd
from IPython.display import clear_output
import io
import os
import glob
import zipfile
import shutil


import numpy as np
import random as python_random
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,BatchNormalization, Dropout,Activation
from tensorflow.keras.metrics import AUC
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from kerastuner.tuners import RandomSearch
from kerastuner import Objective

import sklearn.metrics as sklm
from cxr_foundation import embeddings_data

seed=19
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

#load data
base_path = "C:/STTZ" #Your Path
data_df = pd.read_csv(base_path+"/Extracted_Embeddings/processed_mimic_df.csv")
data_df = data_df[data_df["race"] == 'WHITE']

# #Partition the dataset
df_train = data_df[data_df["split"] == "train"]
df_validate = data_df[data_df["split"] == "validate"]
df_test = data_df[data_df["split"] == "test"]

# #choose same size training sample
df_train = df_train.sample(n=19418, random_state=1)
df_validate = df_validate.sample(n=2523, random_state=1)
df_test = df_test.sample(n=2316, random_state=1)


labels_Columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged_Cardiomediastinum',
        'Fracture','Lung_Lesion','Lung_Opacity','No_Finding','Pleural_Effusion','Pleural_Other',
        'Pneumonia','Pneumothorax','Support_Devices']
# Create training and validation Datasets
training_data = embeddings_data.get_dataset(filenames=df_train.path.values,
                        labels=df_train[labels_Columns].values)

validation_data = embeddings_data.get_dataset(filenames=df_validate.path.values,
                        labels=df_validate[labels_Columns].values)
test_data = embeddings_data.get_dataset(filenames=df_test.path.values,labels=df_test[labels_Columns].values)
#
#
embeddings_size = 1376
num_labels = len(labels_Columns)
batch_size = 32
#
learning_rate = 0.0004244196293577312
dropout_rate1 = 0.5
dropout_rate2 = 0.1
dropout_rate3 = 0.2
dense_units1 = 512
dense_units2 = 256
dense_units3 = 96

early_stopping_callback = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)
#
def build_advanced_model(
        # hp,
         embeddings_size, num_labels):
    # Hyperparameters to tune
    # learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    # # batch_size = hp.Choice('batch_size', values=[16, 32, 64, 128, 256])
    # dropout_rate1 = hp.Float('dropout_rate1', min_value=0.1, max_value=0.5, step=0.1)
    # dropout_rate2 = hp.Float('dropout_rate2', min_value=0.1, max_value=0.5, step=0.1)
    # dropout_rate3 = hp.Float('dropout_rate3', min_value=0.1, max_value=0.5, step=0.1)
    # dense_units1 = hp.Int('dense_units1', min_value=128, max_value=512, step=64)
    # dense_units2 = hp.Int('dense_units2', min_value=64, max_value=256, step=64)
    # dense_units3 = hp.Int('dense_units3', min_value=32, max_value=128, step=32)

    # Define the input layer
    inputs = Input(shape=(embeddings_size,))

    # First hidden layer with L1 and L2 regularization
    x = Dense(dense_units1, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate1)(x)

    # Second hidden layer
    x = Dense(dense_units2, activation='relu')(x)
    x = Dropout(dropout_rate2)(x)

    # Third hidden layer
    x = Dense(dense_units3, activation='relu')(x)
    x = Dropout(dropout_rate3)(x)

    # Output layer for multi-label classification
    outputs = Dense(num_labels, activation='sigmoid')(x)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=[AUC(name='auc', multi_label=True)])

    return model



###Find the best Hyperparameters
# from keras_tuner import RandomSearch
#
# # Define the tuner
# tuner = RandomSearch(
#     lambda hp: build_advanced_model(hp, embeddings_size, num_labels),
#     objective=Objective('val_auc', direction='max'),
#     max_trials=10,
#     executions_per_trial=1,
#     directory='tuner_directory',
#     project_name='ORI42_tuning_project'
# )
#
# tuner.search(
#     x=training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache(),
#     batch_size=batch_size,
#     epochs=20,
#     validation_data=validation_data.batch(batch_size).cache(),
#     callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
# )
#
# # Get the best hyperparameters
# best_hyperparameters = tuner.get_best_hyperparameters()[0]
# print("Best Hyperparameters:")
# print(best_hyperparameters.values)


#

model = build_advanced_model(embeddings_size, num_labels)
# #
# # train the model
history = model.fit(
    x=training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache(),
    validation_data=validation_data.batch(batch_size).cache(),
    callbacks=[early_stopping_callback],
    epochs=50,
)

model.save("W_model.h5")



