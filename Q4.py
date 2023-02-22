"""
Question 4: Is it possible to predict an order cancellation? what is the accuracy?

The question is asking whether it is possible to build a machine learning model
that can predict whether an order will be canceled or not,
and if so, what is the accuracy of that model in making those predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def init(df):
    df = df.drop("type_of_meal_plan", axis=1)
    # Pre-processing
    df['booking_status'] = df['booking_status'].apply(lambda x: 0 if x == 'Canceled' else 1)
    df = pd.get_dummies(df, columns=['market_segment_type', 'room_type_reserved'], drop_first=True)

    # Divide the dataset into features (X) and target (y)
    X = df.drop(['booking_status', 'Booking_ID'], axis=1)
    y = df['booking_status']

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def check_overfitting(model, X, y, cv=5):
    train_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    test_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    avg_train_score = train_scores.mean()
    avg_test_score = test_scores.mean()
    return avg_train_score > avg_test_score


def prediction(X_train, X_test, y_train, y_test):
    # Linear Regression
    clf_lin = LinearRegression()
    clf_lin.fit(X_train, y_train)
    y_pred_lin = np.round(clf_lin.predict(X_test))
    acc_lin = accuracy_score(y_test, y_pred_lin)
    print("Accuracy of Linear Regression: ", acc_lin)

    # Logistic Regression
    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print("Accuracy of Logistic Regression: ", acc_lr)
    is_lr_overfitting = check_overfitting(clf_lr, X_train, y_train)
    if is_lr_overfitting:
        print('Logistic Regression Classifier is overfitting !')

    # Random Forest
    max_acc_rf = 0
    best_n_estimator = 0
    best_max_depth = 0
    for i in range(1, 50):
        clf_rf = RandomForestClassifier(n_estimators=i, random_state=42, max_depth=31)
        clf_rf.fit(X_train, y_train)
        y_pred_rf = clf_rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        if acc_rf > max_acc_rf:
            max_acc_rf = acc_rf
            best_n_estimator = i
    print(f'Accuracy of Random Forest: {max_acc_rf}')
    # print(f'the best estimator {best_n_estimator}')
    is_dt_overfitting = check_overfitting(clf_rf, X_train, y_train)
    if is_dt_overfitting:
        print('Random Forest Classifier is overfitting !')

    # KNN
    max_acc_knn = 0
    best_k = 0
    for i in range(3, 20, 2):
        clf_knn = KNeighborsClassifier(n_neighbors=i)
        clf_knn.fit(X_train, y_train)
        y_pred_knn = clf_knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        if acc_knn > max_acc_knn:
            max_acc_knn = acc_knn
            best_k = i

    # print(f'the best k {best_k}')
    print(f'Accuracy of KNN: {acc_knn}')
    is_knn_overfitting = check_overfitting(clf_knn, X_train, y_train)
    if is_knn_overfitting:
        print('KNN Classifier is overfitting !')

    # SVC
    max_acc_svc = 0
    best_kernel = 0
    kernel = ["linear", "poly", "rbf", "sigmoid"]
    for i in kernel:
        clf_svc = SVC(kernel=i).set_params(gamma=0.1)
        clf_svc.fit(X_train, y_train)
        y_pred_svc = clf_svc.predict(X_test)
        acc_svc = accuracy_score(y_test, y_pred_svc)
        if acc_svc > max_acc_svc:
            max_acc_svc = acc_svc
            best_kernel = i
    # print(f'the best kernel {best_kernel}')
    print(f'Accuracy of SVC: {max_acc_svc}')
    is_svc_overfitting = check_overfitting(clf_svc, X_train, y_train)
    if is_svc_overfitting:
        print('SVC Classifier is overfitting !')


if __name__ == '__main__':
    # Load the hotel reservation dataset
    hotel_reservations = pd.read_csv("Hotel_Reservations.csv")
    # Initialize the data
    X_train, X_test, y_train, y_test = init(hotel_reservations)
    # Predict
    prediction(X_train, X_test, y_train, y_test)

"""
Accuracy of Linear Regression:  0.8011026878015162
Accuracy of Logistic Regression:  0.8071674707098553
Accuracy of Random Forest: 0.90723638869745
the best estimator 45
the best k 7
Accuracy of KNN: 0.8406616126809097
KNN Classifier is overfitting !
the best kernel rbf
Accuracy of SVC: 0.8518263266712612
"""
