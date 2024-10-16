import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv('C:/Users/Yashashwini/Downloads/mini_diabetes.csv')

# Handle missing values if any
data.fillna(data.mean(), inplace=True)

# Split features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#-------------------------------------------------------------------------
# Initialize Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# Fit the model
dt.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Accuracy: {accuracy_dt}')
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

#--------------------------------------------------------------------------
# Initialize KNN with GridSearchCV for hyperparameter tuning
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 50)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(X_train, y_train)

# Best parameters and KNN model
print(f'Best Parameters for KNN: {knn_gscv.best_params_}')
knn_best = knn_gscv.best_estimator_

# Predict on the test set
y_pred_knn = knn_best.predict(X_test)

# Evaluate the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy_knn}')
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

#---------------------------------------------------------------------------
# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Fit the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

#---------------------------------------------------------------------------
# Initialize AdaBoost
ada = AdaBoostClassifier(random_state=42)

# Fit the model
ada.fit(X_train, y_train)

# Predict on the test set
y_pred_ada = ada.predict(X_test)

# Evaluate the model
accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(f'AdaBoost Accuracy: {accuracy_ada}')
print(confusion_matrix(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))

#--------------------------------------------------------------------------
# Initialize BaggingClassifier with Decision Tree
bagging_dt = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)

# Fit the model
bagging_dt.fit(X_train, y_train)

# Predict on the test set
y_pred_bagging_dt = bagging_dt.predict(X_test)

# Evaluate the model
accuracy_bagging_dt = accuracy_score(y_test, y_pred_bagging_dt)
print(f'Bagging Decision Tree Accuracy: {accuracy_bagging_dt}')
print(confusion_matrix(y_test, y_pred_bagging_dt))
print(classification_report(y_test, y_pred_bagging_dt))

#---------------------------------------------------------------------------
# Initialize StackingClassifier with multiple classifiers
estimators = [
    ('dt', dt),
    ('knn', knn_best),
    ('rf', rf),
    ('ada', ada)
]
stacking = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(random_state=42))

# Fit the model
stacking.fit(X_train, y_train)

# Predict on the test set
y_pred_stacking = stacking.predict(X_test)

# Evaluate the model
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f'Stacking Accuracy: {accuracy_stacking}')
print(confusion_matrix(y_test, y_pred_stacking))
print(classification_report(y_test, y_pred_stacking))

print(f"Decision Tree Accuracy: {accuracy_dt}")
print(f"KNN Accuracy: {accuracy_knn}")
print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"AdaBoost Accuracy: {accuracy_ada}")
print(f"Bagging Decision Tree Accuracy: {accuracy_bagging_dt}")
print(f"Stacking Accuracy: {accuracy_stacking}")
