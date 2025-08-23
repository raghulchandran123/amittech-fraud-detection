# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

print("Loading data for training...")
# We'll train on the first version of the data
df = pd.read_csv('data/v0/transactions_2022.csv')

# Define features (X) and target (y)
# The dataset has anonymized features V1-V28 and the Amount
features = [f'V{i}' for i in range(1, 29)] + ['Amount']
target = 'Class'

X = df[features]
y = df[target]

# Split data for training and validation as instructed [cite: 27]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train the model
# Using class_weight='balanced' is helpful for imbalanced datasets [cite: 11]
model = LogisticRegression(class_weight='balanced', max_iter=1000)
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
preds = model.predict(X_test)
f1 = f1_score(y_test, preds)
print(f"Model F1-score on test set: {f1:.4f}")

# Save the trained model to a file
joblib.dump(model, 'fraud_model.joblib')
print("Model saved as fraud_model.joblib")
