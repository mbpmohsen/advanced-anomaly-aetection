from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Read the CSV file using pandas
df = pd.read_csv('./preprocessed_nasa_logs.csv')

# Step 3: Prepare the data
# Select numerical features (including TF-IDF and scaled features)
features = df.select_dtypes(include=['float64', 'int64']).columns
X = df[features]

# Step 4: Train-Test Split (optional for unsupervised learning)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Step 5: Train the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# -1 indicates an anomaly, 1 indicates normal data
df_test_results = X_test.copy()
df_test_results['anomaly'] = y_pred

# Step 7: Display the anomalies
anomalies = df_test_results[df_test_results['anomaly'] == -1]
print(f"Number of anomalies detected: {len(anomalies)}")
print(anomalies)

# Optionally, save the anomalies to a CSV file
anomalies.to_csv('detected_anomalies.csv', index=False)