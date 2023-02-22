"""
Question 1: What factors most influence the booking status (cancelled or not cancelled)?

This question seeks to identify the most significant factors that affect whether a booking is canceled or not.
It involves analyzing the impact of various factors such as the lead time, the number of adults and children,
the type of meal, the deposit type, and other features on the likelihood of a booking being canceled or not.
The goal is to identify the features that have the most significant impact on the booking status
and to build a model that can predict the likelihood of a booking being canceled based on these factors.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def showChart(importance):
    features = []
    scores = []
    # Print the feature importance
    for i, v in enumerate(importance):
        features.append(columns[i])
        scores.append(v)
        print('Feature: %0d, Score: %.5f, Feature Name: %s' % (i, v, columns[i]))
    print('\n')
    plt.figure(figsize=(10, 8))
    plt.bar(features, scores)
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance')
    plt.show()


if __name__ == '__main__':
    # Read the CSV file into a DataFrame
    df = pd.read_csv("Hotel_Reservations.csv")

    # drop the 'Booking_ID' column
    df = df.drop(['Booking_ID'], axis=1)

    # one-hot encode the 'room_type_reserved' column
    df = pd.get_dummies(df,
                        columns=['room_type_reserved', 'type_of_meal_plan', 'market_segment_type', 'repeated_guest'])

    # Extract the features and labels from the DataFrame
    X = df.drop(["booking_status"], axis=1)
    y = df["booking_status"]

    # Saving column names before splitting the data
    columns = X.columns

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert X_train and X_test back to dataframe after splitting and then use the columns attribute
    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)

    # Create an instance of the RandomForestClassifier and DecisionTreeClassifier classes
    classifier = RandomForestClassifier()
    classifier_dt = DecisionTreeClassifier()

    # Fit the classifiers to the training data
    classifier.fit(X_train, y_train)
    classifier_dt.fit(X_train, y_train)

    # Check the feature importance
    importance = classifier.feature_importances_
    importance_dt = classifier_dt.feature_importances_

    # show Results
    showChart(importance)
    showChart(importance_dt)
