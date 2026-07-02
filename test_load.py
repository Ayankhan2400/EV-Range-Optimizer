import joblib
import pandas as pd

# 1. Load the frozen model directly from the file disk
loaded_brain = joblib.load('ev_model.pkl')
print(" Success: The model file was loaded successfully!")

# 2. Create a random new trip leg to test it
test_leg = pd.DataFrame([{'Speed_kmh': 80, 'Temperature_C': 25, 'Elevation_Change_m': 10, 'Driving_Style_Score': 1.2}])

# 3. Predict using the loaded asset
prediction = loaded_brain.predict(test_leg)[0]
print(f" The loaded AI predicted a consumption rate of: {prediction:.2f} Wh/km")