
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Load the dataset
data = pd.read_csv('breastCancerData.csv')

# Separate features (X) and target variable (y)
X = data.iloc[:, 2:]  # Exclude ID column and use all numeric columns for features
y = data['diagnosis']  # Target variable

# Convert the target variable to binary numerical values
y = y.map({'M': 1, 'B': 0})

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix (Decision Tree)')
plt.show()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy Decision Tree:", accuracy)

# Train the Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix (Random Forest)')
plt.show()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy Random Forest:", accuracy)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the deep neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                    epochs=10, batch_size=64, callbacks=[early_stopping])

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix (Neural Network)')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA - Breast Cancer Data')
# plt.show()

# # Apply ICA to visualize feature importance
# ica = FastICA(n_components=2)
# X_ica = ica.fit_transform(X)

# plt.scatter(X_ica[:, 0], X_ica[:, 1], c=y, cmap='coolwarm')
# plt.xlabel('Independent Component 1')
# plt.ylabel('Independent Component 2')
# plt.title('ICA - Breast Cancer Data')
# plt.show()

fig, axes = plt.subplots(1, 2)

# Plot before PCA
axes[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Before PCA')

# Apply PCA to visualize feature importance
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot after PCA
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y)
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].set_title('After PCA')

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 2)

# Plot before ICA
axes[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Before ICA')

# Apply ICA to visualize feature importance
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X)

# Plot after ICA
axes[1].scatter(X_ica[:, 0], X_ica[:, 1], c=y)
axes[1].set_xlabel('Independent Component 1')
axes[1].set_ylabel('Independent Component 2')
axes[1].set_title('After ICA')

plt.tight_layout()
plt.show()

