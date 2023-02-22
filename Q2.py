"""
Question 2: How does the number of adults, children, and special requests affect the average price per room?

The question is asking to analyze the relationship between the number of adults, children, and special requests
and the average price per room.
The results from the Linear Regression, Random Forest, and Decision Tree algorithms provide predictions
for the average price per room based on the number of adults, children, and special requests.
These predictions can be used to understand how changes in the number of adults,
children, and special requests affect the average price per room.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def showChart(hotel_reservations, factor):
    # Compute the average price per room for each value of no_of_adults
    price_by_adults = hotel_reservations.groupby(factor)["avg_price_per_room"].mean()

    # Create a bar chart of the average price per room by number of adults
    plt.bar(price_by_adults.index, price_by_adults.values)
    plt.xlabel(factor)
    plt.ylabel("Average Price per Room (€)")
    plt.show()


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv("Hotel_Reservations.csv")

    # Prepare the data for training
    X = df[['no_of_adults', 'no_of_children', 'no_of_special_requests']]
    y = df['avg_price_per_room']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the models
    lr = LinearRegression()
    rf = RandomForestRegressor()
    dt = DecisionTreeRegressor()

    # Train the models
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_dt = dt.predict(X_test)

    # Evaluate the models using mean absolute error
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)

    print("Mean Absolute Error of Linear Regression: ", mae_lr)
    print("Mean Absolute Error of Random Forest: ", mae_rf)
    print("Mean Absolute Error of Decision Tree: ", mae_dt)
    print()
    print("Linear Regression predictions: ", y_pred_lr)
    print("Random Forest predictions: ", y_pred_rf)
    print("Decision Tree predictions: ", y_pred_dt)

    # Create a figure
    fig, ax = plt.subplots()

    # Add the actual values of average_daily_rate to the plot
    ax.scatter(range(len(y_test)), y_test, label='Actual Values')

    # Add the predictions of each model to the plot
    ax.scatter(range(len(y_pred_lr)), y_pred_lr, label='Linear Regression Predictions')
    ax.scatter(range(len(y_pred_rf)), y_pred_rf, label='Random Forest Predictions')
    ax.scatter(range(len(y_pred_dt)), y_pred_dt, label='Decision Tree Predictions')

    # Add a legend to the plot
    ax.legend()

    # Add x and y axis labels
    plt.xlabel('Index of Observation')
    plt.ylabel('Average Daily Rate')

    # Show the plot
    plt.show()

    # Filter out canceled bookings
    df = df[df["booking_status"] != "Canceled"]

    showChart(df, 'no_of_adults')
    showChart(df, 'no_of_children')
    showChart(df, 'no_of_special_requests')

    """
    Mean Absolute Error part:
    The mean absolute error (MAE) is a measure of how well the model is able to predict the target variable.
    The smaller the MAE, the better the model is at making predictions.
    
    For example, the MAE for the Linear Regression model is 22.93, for the Random Forest model is 21.81,
    and for the Decision Tree model is 21.81.
    
    In this case,
    the Random Forest and Decision Tree models are performing slightly better than the Linear Regression model, with lower MAE.
    It's also worth noting that the difference between the MAE of the three models is not that big.
    
    The predictions presented in ta graph:
    The graph visualizes the relationship between the predicted room prices and the actual room prices.
    From the graph, you can determine the accuracy of each prediction technique by comparing the actual room prices
    and the predicted room prices. If the actual room prices are close to the predicted room prices,
    then the prediction is accurate. However, if the actual room prices are far from the predicted room prices,
    then the prediction is not accurate.
    """
