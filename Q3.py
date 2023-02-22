"""
Question 3: What is the impact of the number of previous cancellations on the booking status?

The question is asking how the number of previous cancellations
for a particular booking affects the probability of that booking being canceled.
It seeks to understand whether there is a correlation between previous cancellations
and the likelihood of a future cancellation, and if so, how strong that correlation is.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the data
df = pd.read_csv("Hotel_Reservations.csv")
df['booking_status'] = df['booking_status'].apply(lambda x: 0 if x == 'Canceled' else 1)

# Prepare the data
X = df[['no_of_previous_cancellations']]
y = df['booking_status']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred_rf = clf.predict(X_test)
print("Accuracy of Random Forest:", accuracy_score(y_test, y_pred_rf))

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("Accuracy of Logistic Regression:", accuracy_score(y_test, y_pred_lr))

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_dt = tree.predict(X_test)
print("Accuracy of Decision Tree:", accuracy_score(y_test, y_pred_dt))

# KNN
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
print("Accuracy of KNN:", accuracy_score(y_test, y_pred_knn))

