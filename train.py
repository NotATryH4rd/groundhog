import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load and preprocess data (as in the provided model code)
df1 = pd.read_csv("data_core.csv")
df2 = pd.read_csv("dataset.csv")

rename_map = {
    'Nitrogen': 'N', 'nitrogen': 'N',
    'Phosphorus': 'P', 'Phosphorous': 'P',
    'Potassium': 'K', 'potassium': 'K',
    'Temparature': 'Temperature', 'temperature': 'Temperature',
    'moisture': 'Moisture', 'Moisture': 'Moisture',
    'pH': 'ph', 'PH': 'ph', 'Ec': 'EC', 'ec': 'EC'
}

df1.rename(columns=lambda x: rename_map.get(x.strip(), x.strip()), inplace=True)
df2.rename(columns=lambda x: rename_map.get(x.strip(), x.strip()), inplace=True)

input_features_1 = ['Temperature', 'Moisture']
input_features_2 = ['ph', 'EC']
target_features = ['N', 'P', 'K']

df1 = df1.dropna(subset=input_features_1 + target_features)
df2 = df2.dropna(subset=input_features_2 + target_features)

X1 = df1[input_features_1]
y1 = df1[target_features]
X2 = df2[input_features_2]
y2 = df2[target_features]

scaler_X1 = StandardScaler()
X1_scaled = scaler_X1.fit_transform(X1)
scaler_X2 = StandardScaler()
X2_scaled = scaler_X2.fit_transform(X2)

target_scaler_1 = StandardScaler()
y1_scaled = target_scaler_1.fit_transform(y1)
target_scaler_2 = StandardScaler()
y2_scaled = target_scaler_2.fit_transform(y2)

X1_train, _, y1_train, _ = train_test_split(X1_scaled, y1_scaled, test_size=0.3, random_state=42)
X2_train, _, y2_train, _ = train_test_split(X2_scaled, y2_scaled, test_size=0.3, random_state=42)

# Build and train models
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(target_features))
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model1 = build_model(X1_train.shape[1])
model2 = build_model(X2_train.shape[1])

model1.fit(X1_train, y1_train, epochs=50, validation_split=0.2, verbose=0)
model2.fit(X2_train, y2_train, epochs=50, validation_split=0.2, verbose=0)

# Save the models and scalers
model1.save("model1.keras")
model2.save("model2.keras")
joblib.dump(scaler_X1, "scaler_X1.pkl")
joblib.dump(scaler_X2, "scaler_X2.pkl")
joblib.dump(target_scaler_1, "target_scaler_1.pkl")
joblib.dump(target_scaler_2, "target_scaler_2.pkl")

print("✅ Models and scalers saved locally!")