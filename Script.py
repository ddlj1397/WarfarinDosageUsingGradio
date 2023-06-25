!pip install --upgrade xlrd
%cd '/content/drive/MyDrive/Intro to Machine Learning'
!ls
# Import section

import os
cwd = os.getcwd()
print(cwd)
import pandas as pd
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
from sklearn.preprocessing import *
from sklearn.model_selection import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import *

Final = pd.read_excel('processed_final_project_data.xlsx')

Final.head(13)

Final.to_excel("FinalDoc.xlsx")

Final.info()

#dropping null values
Final = Final.dropna()

#converting simple categorical data to numeric
Final['Gender'].replace(['male','female'], [0,1], inplace=True)
Final['Age'].replace(['10 - 19','20 - 29','30 - 39','40 - 49','50 - 59','60 - 69','70 - 79','80 - 89','90+'], [0,1,2,3,4,5,6,7,8], inplace=True)
Final['Cyp2C9 genotypes'].replace(['*1/*1','*1/*11','*1/*13','*1/*14','*1/*2','*1/*3','*1/*5','*1/*6','*2/*2','*2/*3','*3/*3'], [1,11,13,14,2,3,5,6,7,8,9],inplace=True)
Final['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'].replace(['A/A','A/G','G/G'], [0,1,2], inplace=True)

#converting race feature into binary data
race_dummies = pd.get_dummies(Final['Race (Reported)'])
Final = pd.concat([Final, race_dummies], axis='columns')
Final = Final.drop(['Race (Reported)'], axis='columns')

Final.info()

# Separate out the features and targets
features = Final.drop(columns='Therapeutic Dose of Warfarin')
targets = pd.DataFrame(Final['Therapeutic Dose of Warfarin'])
targets[targets<30] = 0
targets[targets>=30] = 1

features.info()

# Start of Classification model, first by splitting the data into the train and test set seperated by X and Y

x_train, x_test,y_train, y_test = train_test_split(features, targets, test_size= 0.3, random_state = 42)

# Logistic regression. Algorithm 1
logistic_model = LogisticRegression(random_state = 0)
logistic_model.fit(x_train, y_train)
print("Logistic score:", logistic_model.score(x_test,y_test))
LogScore = logistic_model.predict(x_test)

# Decision Tree. Algorithm 2
from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier(max_depth = 2.5) # define tree model
decision_tree_model.fit(x_train, y_train)
tree_pred = decision_tree_model.predict(x_test)
print('Tree Score:', decision_tree_model.score(x_test, y_test))

# KNN. Algorithm 3
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 5, metric = 'manhattan')
knn_model.fit(x_train, y_train)
print('KNN Score:', knn_model.score(x_test,y_test))
knn_pred = knn_model.predict(x_test)

# Deep learning Keras model. Algorithm 4
model = Sequential()
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
train_history = model.fit(x_train, y_train, validation_data = (x_test, y_test),epochs=10)
print('Keras model score:',model.evaluate(x_test, y_test))
Deepkeras = model.predict(x_test)

# Plot training & validation loss values
plt.plot(train_history.history['loss'], label='Train')
plt.plot(train_history.history['val_loss'], label='Validation')
plt.title('Training & Validation Loss', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(loc='upper right', fontsize=12)

# Plot training & validation accuracy values
plt.plot(train_history.history['accuracy'], label='Train')
plt.plot(train_history.history['val_accuracy'], label='Validation')
plt.title('Training & Validation accuracy', fontsize=15)
plt.ylabel('accuracy', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.show()

# Start of accuracy, precision, recall, F1 and ROC curve scores.

# Logistic scores
logistic_acc = accuracy_score(y_test, LogScore )
logistic_prec = precision_score(y_test, LogScore )
logistic_recall = recall_score(y_test, LogScore )
logistic_roc = roc_auc_score(y_test, LogScore )
logistic_f1 = f1_score(y_test, LogScore )
print("accuracy:", logistic_acc)
print("precision:", logistic_prec)
print("recall:", logistic_recall)
print("roc:", logistic_roc)
print("f1:", logistic_f1)

# KNN Scores
knn_acc = accuracy_score(y_test, knn_pred )
knn_prec = precision_score(y_test, knn_pred )
knn_recall = recall_score(y_test, knn_pred)
knn_roc = roc_auc_score(y_test, knn_pred )
knn_f1 = f1_score(y_test, knn_pred )
print("accuracy:", knn_acc)
print("precision:", knn_prec)
print("recall:", knn_recall)
print("roc:", knn_roc)
print("f1:", knn_f1)

# Decision Tree Scores

tree_acc = accuracy_score(y_test, tree_pred )
tree_prec = precision_score(y_test, tree_pred )
tree_recall = recall_score(y_test, tree_pred)
tree_roc = roc_auc_score(y_test, tree_pred )
tree_f1 = f1_score(y_test, tree_pred )
print("accuracy:", tree_acc)
print("precision:", tree_prec)
print("recall:", tree_recall)
print("roc:", tree_roc)
print("f1:", tree_f1)

Deep_acc = accuracy_score(y_train, Deepkeras)
Deep_prec = precision_score(y_test, Deepkeras)
Deep_recall = recall_score(y_test, Deepkeras)
Deep_roc = roc_auc_score(y_test, Deepkeras)
Deep_f1 = f1_score(y_test, Deepkeras)
print("accuracy:", Deep_acc)
print("precision:", Deep_prec)
print("recall:", Deep_recall)
print("roc:", Deep_roc)
print("f1:", Deep_f1)

import gradio as gr
from joblib import dump, load
import tensorflow as tf
from tensorflow import keras

#loading models
log = load('logistic_model.joblib')
knn = load('knn_model.joblib')
decision = load('decision_tree_model.joblib')
deep = tf.keras.models.load_model('deep_model')

#input/output modules
input_module1 = gr.Dropdown(choices=["Logistic Regression", "KNN", "Decision Tree","Neural Network"], label = "method")
input_module2 = gr.Dropdown(choices=["male", 'female'], label = "gender")
input_module3 = gr.Dropdown(choices=['african-american','asian','black','black african','black or african american',\
                                     'caucasian', 'han chinese', 'hispanic', 'intermediate', 'japanese', 'korean', 'other',\
                                     'Other (Black British)','Other Mixed Race','White'], label = "race")
input_module4 = gr.Number(label='age')
input_module5 = gr.Number(label='height (cm)')
input_module6 = gr.Number(label='weight (kg)')
input_module7 = gr.Checkbox(label='Diabetes')
input_module8 = gr.Checkbox(label='Simvastatin (Zocor)')
input_module9 = gr.Checkbox(label='Amiodarone (Cordarone)')
input_module10 = gr.Number(label='target INR')
input_module11 = gr.Number(label='INR on Reported Therapeutic Dose of Warfarin')
input_module12 = gr.Number(label='Cyp2C9 genotypes (1-13)')
input_module13 = gr.Dropdown(choices=['A/A','A/G','G/G'], label = "VKORC1 genotype")

output_module = gr.Number(label='Therapeutic Dose of Warfarin (>=30 mg/wk) (1=true, 0=false)')

race_options = ['african-american','asian','black','black african','black or african american',\
                                     'caucasian', 'han chinese', 'hispanic', 'intermediate', 'japanese', 'korean', 'other',\
                                     'Other (Black British)','Other Mixed Race','White']

#gradio function
def predict(method, gender, race, age, height, weight, diabetes, simv, amio, targetINR, INR, cyp2c9, vkorc1):

  #converting inputs into numeric data
  if gender == 'male':
    gender = 0
  else:
    gender = 1

  for i in range(len(race_options)):
    if race_options[i] == race:
      race_options[i] = 1
    else:
      race_options[i] = 0
  
  if diabetes == True:
    diabetes = 1
  else:
    diabetes = 0
  
  if simv == True:
    simv = 1
  else:
    simv = 0
  
  if amio == True:
    amio = 1
  else:
    amio = 0

  if vkorc1 == 'A/A':
    vkorc1 = 0
  elif vkorc1 == 'A/G':
    vkorc1 = 1
  else:
    vkorc1 = 2

  #compiling data
  data = [gender, age, height, weight, diabetes, simv, amio, targetINR, INR, cyp2c9, vkorc1]
  data.extend(race_options)
  data.extend([0]) #accounting for extra unused race category (there are 2 'other' options for race)

  #predicting using given method
  if method == "Logistic Regression":
    value = log.predict([data])[0]
  elif method == "KNN":
    value = knn.predict([data])[0]
  elif method == "Decision Tree":
    value = decision.predict([data])[0]
  else:
    value = deep.predict([data])[0]
    if value >= 0.5:
      value = 1
    else:
      value = 0
    

  return value



gr.Interface(fn=predict, inputs=[input_module1,input_module2,input_module3,input_module4,input_module5,input_module6,input_module7,\
                                 input_module8,input_module9,input_module10,input_module11,input_module12,input_module13], outputs=output_module).launch(debug=True)
