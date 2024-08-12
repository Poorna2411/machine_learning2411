import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create the dataset
data = {
    'Feature 1': [1, 2, 3, 4, 5],
    'Feature 2': [2, 3, 5, 7, 9],
    'Label': ['A', 'A', 'B', 'B', 'B']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split features and labels
X = df[['Feature 1', 'Feature 2']]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
