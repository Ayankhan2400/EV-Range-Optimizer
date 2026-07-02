# Importing the core AI libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , r2_score

#telling pandas library to display all columns without hiding any in the middle
pd.set_option('display.max_columns',None)

#telling pandas to expand width so it doesnt wrap lines easily
pd.set_option('display.width',1000)

print("LIBRARIES IMPORTED SUCCESFULLY")

# STEP 3: CREATING THE SYNTHETIC VEHICLE DATA

# 1. Setting a random seed for reproducibility
np.random.seed(42)

# 2. Defining the amount of driving data to generate
num_samples = 5000

# 3. Generating the independent features (environmental and driving factors)
speed = np.random.uniform(20, 120, num_samples)          # Speed in km/h
temperature = np.random.uniform(0, 45, num_samples)      # Temperature in °C
elevation_change = np.random.uniform(-50, 50, num_samples) # Elevation change in meters
driving_style = np.random.uniform(1.0, 2.0, num_samples)  # 1.0=Smooth, 2.0=Aggressive

# 4. Applying physics equations to calculate our target variable (Energy Consumed in Wh/km)
base_consumption = 150 
speed_effect = 0.05 * (speed - 60)**2 
temp_effect = 0.8 * (temperature - 22)**2  
elevation_effect = 2.5 * elevation_change  
style_effect = 30 * (driving_style - 1.0)  

# 5. Adding real-world randomness (noise) to the physics calculations
noise = np.random.normal(0, 10, num_samples)

# 6. Calculating the final target values
energy_consumed_wh_km = base_consumption + speed_effect + temp_effect + elevation_effect + style_effect + noise

# 7. Packaging all our individual arrays into a structured tabular DataFrame
dataset = pd.DataFrame({
    'Speed_kmh': speed,
    'Temperature_C': temperature,
    'Elevation_Change_m': elevation_change,
    'Driving_Style_Score': driving_style,
    'Energy_Consumed_Wh_km': energy_consumed_wh_km
})

print("\n Step 3 Complete: Synthetic vehicle dataset created!")
print(dataset.head(3)) # Prints the first 3 rows of data to the terminal

# STEP 4: SEPARATING FEATURES, TARGETS, AND SPLITTING DATA


# 1. Separate the independent variables (Features) from the dependent variable (Target)
X = dataset[['Speed_kmh', 'Temperature_C', 'Elevation_Change_m', 'Driving_Style_Score']]
y = dataset['Energy_Consumed_Wh_km']

# 2. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Step 4 Complete: Data prepared and partitioned!")
print(f" Training Set Size: {X_train.shape[0]} samples")
print(f" Testing Set Size: {X_test.shape[0]} samples")

#STEP 5 : Initialization and training the AI model

#creating the instance of random forest regressor model
model = RandomForestRegressor(n_estimators=100 , random_state=42)

#train the model using our training data 
print("\n training the random forest model.....")
model.fit(X_train,y_train)

print("\n STEP 5 Complete : AI model has been trained successfully!")

#STEP 6 : evaluating the AI model on unseen data 

#Ask the AI to guess energy consumption for test exam questions
predictions = model.predict(X_test)

#2. Score the AI by comparing its guesses against the hidden true answers
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- MODEL ACCURACY REPORT ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Wh/km")
print(f"Model R² (R-squared) Score: {r2 * 100:.2f}%")
print("--------------------------------")

# ==========================================
# STEP 7: MULTI-LEG TRIP SIMULATION ENGINE
# ==========================================

def simulate_ev_trip(starting_battery_kwh, total_battery_capacity_kwh, trip_route):
    """
    Simulates a multi-segment road trip, tracking battery depletion dynamically 
    using our trained Machine Learning model.
    """
    # 1. Convert starting battery from kWh to Wh (since our model outputs Wh/km)
    current_battery_wh = starting_battery_kwh * 1000  
    safety_threshold_wh = total_battery_capacity_kwh * 1000 * 0.10 # 10% safety margin
    
    total_distance = 0
    print("\n===============================================")
    print(" TRIP SIMULATION ENGINE ACTIVATED")
    print(f" Starting Battery: {starting_battery_kwh} kWh")
    print("===============================================")
    
    # 2. Loop through every single leg of the trip
    for index, leg in enumerate(trip_route):
        # Package this single leg's features into a Pandas DataFrame format matching the model
        leg_data = pd.DataFrame([{
            'Speed_kmh': leg['speed'],
            'Temperature_C': leg['temp'],
            'Elevation_Change_m': leg['elevation'],
            'Driving_Style_Score': leg['style']
        }])
        
        # 3. AI predicts the consumption rate (Wh/km) for this specific stretch of road
        predicted_rate = model.predict(leg_data)[0]
        
        # 4. Calculate total energy used in this leg: Rate (Wh/km) * Distance (km)
        energy_used_wh = predicted_rate * leg['distance']
        
        # Deduct the energy used from our car's battery bank
        current_battery_wh -= energy_used_wh
        total_distance += leg['distance']
        
        print(f" Leg {index + 1} ({leg['name']}):")
        print(f" Conditions: {leg['speed']} km/h, {leg['temp']}°C, {leg['elevation']}m elevation change")
        print(f" Predicted Drain Rate: {predicted_rate:.1f} Wh/km")
        print(f" Remaining Battery: {max(0, current_battery_wh / 1000):.2f} kWh\n")
        
        # 5. Emergency Break Condition: Did we completely run out of juice mid-trip?
        if current_battery_wh <= 0:
            print(f" CRITICAL FAILURE: Your EV ran out of charge at kilometer {total_distance} during the '{leg['name']}' leg!")
            print(" Recommendation: Rerouting required! Add a charging station stop immediately.")
            return

    # 6. Final safety evaluation after completing all legs
    final_battery_kwh = current_battery_wh / 1000
    print("===============================================")
    print(f" DESTINATION REACHED! Total Distance: {total_distance} km")
    print(f" Final Battery Level: {final_battery_kwh:.2f} kWh")
    
    if current_battery_wh < safety_threshold_wh:
        print(" RANGE ANXIETY WARNING: You arrived safely, but your battery is below the 10% safety margin.")
        print(" Advice: Plug your vehicle into a charger immediately.")
    else:
        print(" SUCCESS: Optimal trip profile. You have plenty of safety reserve left.")
    print("===============================================\n")

# --- TEST SCENARIO: Let's run a live road trip simulation ---
# Imagine a user driving a vehicle with a 40 kWh battery pack, starting with 25 kWh of juice.
sample_route = [
    {'name': 'City Traffic Grid', 'distance': 12, 'speed': 35, 'temp': 38, 'elevation': 0, 'style': 1.1},
    {'name': 'Expressway Speedways', 'distance': 50, 'speed': 110, 'temp': 32, 'elevation': 10, 'style': 1.8},
    {'name': 'Mountain Downhill Pass', 'distance': 30, 'speed': 60, 'temp': 20, 'elevation': -45, 'style': 1.2}
]

simulate_ev_trip(starting_battery_kwh=25, total_battery_capacity_kwh=40, trip_route=sample_route)

#STEP 8 : Saving the model into a File

import joblib

joblib.dump(model, 'ev_model.pkl')

print("\n STEP 8 Complete : trained AI model saved as ev_model.pkl") 