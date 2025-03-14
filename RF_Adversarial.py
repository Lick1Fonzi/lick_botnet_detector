import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#read original Rbot dataset 
data = pd.read_csv("data/raw/Rbot.csv")
training_set = data.copy(deep=True)
training_set['IPSrcType'] = np.where(training_set["Src IP"].str.startswith("147.32."), 1, 0)
training_set['IPDstType'] = np.where(training_set["Dst IP"].str.startswith("147.32."), 1, 0)
training_set = training_set.drop(columns=['Src IP', 'Dst IP'])


X_train, X_test, y_train, y_test = train_test_split(training_set.drop(columns=['Label']), training_set.Label, test_size = 0.25, random_state = 42, stratify = training_set.Label)

model = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state=42)

# Train the model with *fit()* function on the training set.
model.fit(X_train, y_train)

# Predict unseen test data.
pred = model.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, pred)}")
print(f"Recall Score: {recall_score(y_test, pred)}")
print(f"Precision Score: {precision_score(y_test, pred)}")
print(f"F1 Score: {f1_score(y_test, pred)}")


ADVERSARIAL = 0
if(ADVERSARIAL):
    #Prediction with Adversarial Dataset generated from rbot applying perturbations steps
    # Also modify Src and Dst IP attributes into types (if starting with 147.32 or not)
    training_set = pd.read_csv("data/raw/Rbot_adversarial-1a-1.csv")
    training_set['IPSrcType'] = np.where(training_set["Src IP"].str.startswith("147.32."), 1, 0)
    training_set['IPDstType'] = np.where(training_set["Dst IP"].str.startswith("147.32."), 1, 0)
    training_set = training_set.drop(columns=['Src IP', 'Dst IP'])

    labels = training_set.Label

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(training_set.drop(columns=['Label']), labels, test_size = 0.5, random_state = 42, stratify = training_set.Label)

    pred = model.predict(X_test)

    print("Adversarial dataset prediction scores")
    print(f"Accuracy Score: {accuracy_score(y_test, pred)}")
    print(f"Recall Score: {recall_score(y_test, pred)}")
    print(f"Precision Score: {precision_score(y_test, pred)}")
    print(f"F1 Score: {f1_score(y_test, pred)}")