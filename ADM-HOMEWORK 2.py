#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[16]:


# Load the dataset
file_path = "car_evaluation.csv" 
df = pd.read_csv(file_path)

# Shuffle the dataset with random seed 42
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.head()


# In[17]:


# Convert categorical features into numerical values using Label Encoding
encoder = LabelEncoder()
X = df.drop(columns=['label']).apply(encoder.fit_transform)
y = df['label']

X.head()


# In[18]:


# Split into Train (60%), Validation (20%), and Test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display the shape of each split
X_train.shape, X_val.shape, X_test.shape


# In[19]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# In[20]:


# Train an SVM classifier with C=1.0
svm_model = SVC(C=1.0, kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)


# In[21]:


# Get predictions
y_train_pred = svm_model.predict(X_train_scaled)
y_val_pred = svm_model.predict(X_val_scaled)
y_test_pred = svm_model.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# In[22]:


#b.
import numpy as np

# Define the range of C values: {10^-3, 10^-2, ..., 10^2, 10^3}
C_values = np.logspace(-3, 3, num=7)
C_values


# In[23]:




# Lists to store accuracy results
train_accuracies = []
val_accuracies = []
test_accuracies = []

# Train SVM with different values of C
for C in C_values:
    svm_model = SVC(C=C, kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Get predictions
    y_train_pred = svm_model.predict(X_train_scaled)
    y_val_pred = svm_model.predict(X_val_scaled)
    y_test_pred = svm_model.predict(X_test_scaled)

    # Compute accuracy
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    val_accuracies.append(accuracy_score(y_val, y_val_pred))
    test_accuracies.append(accuracy_score(y_test, y_test_pred))

# Print accuracy results for each C value
for i, C in enumerate(C_values):
    print(f"C={C:.4f} | Train Acc: {train_accuracies[i]:.4f} | Val Acc: {val_accuracies[i]:.4f} | Test Acc: {test_accuracies[i]:.4f}")


# In[24]:


import matplotlib.pyplot as plt

# Plot accuracy for different C values
plt.figure(figsize=(8, 6))
plt.plot(C_values, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(C_values, val_accuracies, label="Validation Accuracy", marker="s")
plt.plot(C_values, test_accuracies, label="Test Accuracy", marker="x")

# Log scale for x-axis (C values)
plt.xscale("log")
plt.xlabel("C value (log scale)")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy vs C values")
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


# Find the best C value based on validation accuracy
best_C_index = np.argmax(val_accuracies)
best_C = C_values[best_C_index]

print(f"The best generalization performance is achieved at C={best_C:.4f}, "
      f"with Validation Accuracy: {val_accuracies[best_C_index]:.4f} and Test Accuracy: {test_accuracies[best_C_index]:.4f}.")


# In[30]:


#2 QUESTION 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[31]:


# Load the dataset
file_path = "wine.csv"  # Ensure this file is in the same directory
df = pd.read_csv(file_path)

# Shuffle the dataset with random seed 42
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the first few rows
df.head()


# In[32]:


# Define features (X) and target (y)
X = df.drop(columns=['quality'])  # All columns except 'quality'
y = df['quality']  # Target variable

# Split into Training (70%) and Test (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X_train.shape, X_test.shape


# In[33]:


# Train a Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)


# In[34]:


# Get predictions on the test set
y_pred = dt_model.predict(X_test)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract TP, FP, TN, FN from Confusion Matrix
tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (None, None, None, None)  # Handle multi-class cases
print("Confusion Matrix:\n", conf_matrix)

# Display extracted values if binary classification
if tn is not None:
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")


# In[35]:


# Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[38]:


#B 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[39]:


# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[40]:


# Get predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Compute Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Extract TP, FP, TN, FN from Confusion Matrix (if binary classification)
tn_rf, fp_rf, fn_rf, tp_rf = conf_matrix_rf.ravel() if conf_matrix_rf.size == 4 else (None, None, None, None)

print("Confusion Matrix (Random Forest):\n", conf_matrix_rf)

# Display extracted values if binary classification
if tn_rf is not None:
    print(f"True Positives (TP): {tp_rf}")
    print(f"False Positives (FP): {fp_rf}")
    print(f"True Negatives (TN): {tn_rf}")
    print(f"False Negatives (FN): {fn_rf}")


# In[42]:


# Generate classification report with zero_division handling to prevent warnings
report_rf = classification_report(y_test, y_pred_rf, zero_division=1)
print("Classification Report (Random Forest):\n", report_rf)


# In[ ]:




