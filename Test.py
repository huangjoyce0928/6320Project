from tensorflow.keras.models import load_model
import pandas as pd
from IPython.display import clear_output
import io
import os
import glob
import zipfile
import shutil

import numpy as np
import matplotlib.pyplot as plt
from cxr_foundation import embeddings_data
#load data
base_path = "C:/STTZ"   #Your Path
data_df = pd.read_csv(base_path+"/Extracted_Embeddings/processed_mimic_df.csv")


# Count the number of samples for each race
race_counts = data_df['race'].value_counts()

colors = plt.cm.viridis(np.linspace(0, 1, len(race_counts)))

# Create a bar chart
plt.figure(figsize=(10, 6))
race_counts.plot(kind='bar', color=colors)
plt.xlabel('Race')
plt.ylabel('Number of Samples')
plt.title('Sample Count by Race')
plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees

plt.subplots_adjust(bottom=0.25)
plt.show()

data_df = data_df[data_df["race"] == 'WHITE']
# data_df = data_df[data_df["race"] == 'BLACK/AFRICAN AMERICAN']

# #Partition the dataset
df_train = data_df[data_df["split"] == "train"]
df_validate = data_df[data_df["split"] == "validate"]
df_test = data_df[data_df["split"] == "test"]

labels_Columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged_Cardiomediastinum',
        'Fracture','Lung_Lesion','Lung_Opacity','No_Finding','Pleural_Effusion','Pleural_Other',
        'Pneumonia','Pneumothorax','Support_Devices']
# Create training and validation Datasets
training_data = embeddings_data.get_dataset(filenames=df_train.path.values,
                        labels=df_train[labels_Columns].values)

validation_data = embeddings_data.get_dataset(filenames=df_validate.path.values,
                        labels=df_validate[labels_Columns].values)
test_data = embeddings_data.get_dataset(filenames=df_test.path.values,labels=df_test[labels_Columns].values)


loaded_model_W = load_model("W_model.h5")
loaded_model_B = load_model("B_model.h5")
loaded_model_W5B5 = load_model("W5B5_model.h5")
loaded_model_W7B3 = load_model("W7B3_model.h5")
loaded_model_W3B7 = load_model("W3B7_model.h5")
loaded_model_ORI = load_model("ORIGINAL_model.h5")
embeddings_size = 1376
num_labels = len(labels_Columns)
batch_size = 32
from sklearn.metrics import roc_auc_score, roc_curve, auc
# #
# Evaluate using ROC-AUC
y_pred_prob_W = loaded_model_W.predict(test_data.batch(batch_size))
y_pred_prob_B = loaded_model_B.predict(test_data.batch(batch_size))
y_pred_prob_W5B5 = loaded_model_W5B5.predict(test_data.batch(batch_size))
y_pred_prob_W7B3 = loaded_model_W7B3.predict(test_data.batch(batch_size))
y_pred_prob_W3B7 = loaded_model_W3B7.predict(test_data.batch(batch_size))
y_pred_prob_ORI = loaded_model_ORI.predict(test_data.batch(batch_size))

# # #
##calculate Cardiomegaly AUC in different model
label_name = 'Cardiomegaly' #Change the label you want to calculate
label_index = labels_Columns.index(label_name)
#
# Calculate ROC-AUC for 'No_Finding'
roc_auc_no_finding_W = roc_auc_score(df_test[label_name], y_pred_prob_W[:, label_index])
roc_auc_no_finding_B = roc_auc_score(df_test[label_name], y_pred_prob_B[:, label_index])
roc_auc_no_finding_W5B5 = roc_auc_score(df_test[label_name], y_pred_prob_W5B5[:, label_index])
roc_auc_no_finding_W7B3 = roc_auc_score(df_test[label_name], y_pred_prob_W7B3[:, label_index])
roc_auc_no_finding_W3B7 = roc_auc_score(df_test[label_name], y_pred_prob_W3B7[:, label_index])
roc_auc_no_finding_ORI = roc_auc_score(df_test[label_name], y_pred_prob_ORI[:, label_index])

print(roc_auc_no_finding_W,roc_auc_no_finding_W7B3,roc_auc_no_finding_W5B5
      ,roc_auc_no_finding_W3B7,roc_auc_no_finding_B,roc_auc_no_finding_ORI)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Set color
colors = [(0, 0.5, 0), (1, 0.5, 0)]  # 绿色到橙色
cmap_name = 'my_list'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=5)

width = 0.4
percentages = ['100%', '70%', '50%', '30%', '0%','Original']
auc_scores = [roc_auc_no_finding_W, roc_auc_no_finding_W7B3, roc_auc_no_finding_W5B5,
        roc_auc_no_finding_W3B7, roc_auc_no_finding_B, roc_auc_no_finding_ORI]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(percentages))

for i, auc in enumerate(auc_scores):
    ax.bar(x[i], auc, width, color=cm(i / len(auc_scores)), label=percentages[i])

ax.scatter(x, auc_scores, color='gray', s=50)  #Add connect point
ax.plot(x, auc_scores, color='grey', marker='o')

plt.subplots_adjust(bottom=0.2, left=0.1)
ax.set_ylim(0.80, 0.95)
ax.set_ylabel('AUC', fontsize=14)
ax.set_xlabel('Percentage of White Data in Training Dataset', fontsize=14)
ax.set_title('AUC of No_Finding testing in Black dataset', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(percentages)
plt.show()


# # Plot ROC-AUC curve for each class
# plt.figure(figsize=(10, 7))
# for i, label_name in enumerate(labels_Columns):
#     fpr, tpr, _ = roc_curve(df_test[labels_Columns].values[:, i], y_pred_prob[:, i])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=2, label=f'{label_name} (AUC = {roc_auc:.3f})')
#
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'W-Trained Model on W Dataset (Average ROC-AUC= {average_roc_auc:.3f})')
# plt.legend(loc="lower right")
# # plt.savefig('White-Trained Model on Black Dataset.png')
# plt.show()
