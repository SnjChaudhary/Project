import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

data = pd.read_csv(r"C:\Users\snjch\Desktop\predictive_maintenance_quality_control\data\sensor_data.csv")

data['timestamp'] = pd.to_datetime(data['timestamp'])

data['timestamp'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds()

data['temp_rolling_avg'] = data['temperature'].rolling(window=3).mean()
data['vibration_diff'] = data['vibration'].diff()

data.ffill(inplace=True)

X = data.drop(['failure'], axis=1)
y = data['failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

with open(r'C:\Users\snjch\Desktop\predictive_maintenance_quality_control\models\predictive_model.pkl', 'wb') as f:
    pickle.dump(model, f)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Predictive Maintenance Model Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))

