import streamlit as st
import pandas as pd
import joblib
import requests
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# 1. Page Configuration
st.set_page_config(page_title="AuraCharge EV Optimizer", page_icon="⚡", layout="wide")

# 2. Load our pre-trained AI brain asset (Cached so it stays in RAM)
@st.cache_resource
def load_model():
    return joblib.load('ev_model.pkl')

model = load_model()

# Initialize the open-source geolocator map engine with a generous timeout
geolocator = Nominatim(user_agent="my_unique_ev_range_optimizer_v2", timeout=10)


# ✨ NEW OPTIMIZATION: Cache the city name coordinate lookups
@st.cache_data
def get_cached_coordinates(location_name):
    """Fetches and caches latitude/longitude for a city text string."""
    try:
        return geolocator.geocode(location_name)
    except Exception:
        return None


# ✨ NEW OPTIMIZATION: Cache the heavy network highway routing call
@st.cache_data
def fetch_road_route(start_lat, start_lon, end_lat, end_lon):
    """Queries and caches the OSRM highway road path coordinates."""
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10).json()
        route_geometry = response['routes'][0]['geometry']['coordinates']
        flipped_route = [[point[1], point[0]] for point in route_geometry]
        driving_distance_km = response['routes'][0]['distance'] / 1000
        return flipped_route, driving_distance_km
    except Exception:
        return [[start_lat, start_lon], [end_lat, end_lon]], None


# 3. Main Header
st.title("⚡ AuraCharge AI")
st.subheader("Predictive EV Route Planner & Core Battery Optimizer")
st.markdown("---")

# 4. User Sidebar Layout (Battery Telemetry System)
st.sidebar.header("🔋 Battery Telemetry")
total_capacity = st.sidebar.slider("Total Battery Capacity (kWh)", 30, 100, 60, step=5)
current_charge = st.sidebar.slider("Current Battery Charge (kWh)", 5, total_capacity, 45, step=1)

# Dynamic Battery Percentage Icon Calculation
battery_percentage = (current_charge / total_capacity) * 100

if battery_percentage >= 75:
    battery_icon = "🔋 (High Charge)"
elif battery_percentage >= 30:
    battery_icon = "🪫 (Medium Charge)"
else:
    battery_icon = "🚨 (Low Battery)"

st.sidebar.metric(label=f"State of Charge {battery_icon}", value=f"{battery_percentage:.1f} %")
st.sidebar.markdown("---")

# 5. Main Dashboard Layout splits into 2 visual columns
col_inputs, col_map = st.columns([1, 1])

driving_distance = 0.0
road_lines = []

with col_inputs:
    st.header("📍 Route & Environmental Parameters")
    
    # Destination Input Fields
    departure_loc = st.text_input("Departure Location", "Lucknow")
    arrival_loc = st.text_input("Arrival Location", "Kanpur")
    
    st.markdown("### ☁️ Live Ambient Conditions")
    speed = st.slider("Target Cruising Speed (km/h)", 20, 120, 80)
    temp = st.slider("Ambient Temperature (°C)", 0, 45, 25)
    elevation = st.slider("Net Elevation Change (meters)", -50, 50, 0, step=5)
    
    style = st.select_slider(
        "Driving Acceleration Profile",
        options=[1.0, 1.3, 1.6, 2.0],
        value=1.3,
        format_func=lambda x: "Eco Smooth" if x==1.0 else ("Moderate" if x==1.3 else ("Sporty" if x==1.6 else "Aggressive Heavy"))
    )

# 6. Advanced Road Routing & Map Generation
with col_map:
    st.header("🗺️ Live Road Guidance Navigation")
    
    # Use our optimized cached function calls instead of direct raw calls
    dep_geo = get_cached_coordinates(departure_loc)
    arr_geo = get_cached_coordinates(arrival_loc)
    
    if dep_geo and arr_geo:
        road_lines, driving_distance = fetch_road_route(
            dep_geo.latitude, dep_geo.longitude, 
            arr_geo.latitude, arr_geo.longitude
        )
        
        # Center the map canvas right between the two locations
        center_lat = (dep_geo.latitude + arr_geo.latitude) / 2
        center_lon = (dep_geo.longitude + arr_geo.longitude) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
        
        # Drop Pins
        folium.Marker([dep_geo.latitude, dep_geo.longitude], popup=departure_loc, tooltip="Start", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([arr_geo.latitude, arr_geo.longitude], popup=arrival_loc, tooltip="Destination", icon=folium.Icon(color='red')).add_to(m)
        
        # Draw Highway route tracking line
        if road_lines:
            folium.PolyLine(road_lines, color="#3498db", weight=6, opacity=0.85, tooltip="Calculated EV Highway Path").add_to(m)
        
        # Render map interface visual element
        st_folium(m, width=650, height=400, key=f"map_{departure_loc}_{arrival_loc}")
        
        if driving_distance:
            st.info(f"🛣️ **Verified Highway Distance:** {driving_distance:.2f} km (Fetched instantly via cache memory)")
    else:
        st.warning("Waiting for valid location inputs...")

st.markdown("---")

# 7. AI Analysis Trigger Execution
if st.button("🔮 Run Route Optimization Analysis", use_container_width=True):
    if driving_distance == 0.0:
        st.error("Invalid route. Please enter valid locations before analyzing.")
    else:
        input_data = pd.DataFrame([{
            'Speed_kmh': speed,
            'Temperature_C': temp,
            'Elevation_Change_m': elevation,
            'Driving_Style_Score': style
        }])
        
        predicted_wh_km = model.predict(input_data)[0]
        total_energy_used_kwh = (predicted_wh_km * driving_distance) / 1000
        remaining_battery_kwh = current_charge - total_energy_used_kwh
        safety_limit = total_capacity * 0.10
        
        st.markdown("### 📊 AI Optimization Report")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        metrics_col1.metric("Predicted Drain Rate", f"{predicted_wh_km:.1f} Wh/km")
        metrics_col2.metric("Total Route Energy Cost", f"{total_energy_used_kwh:.2f} kWh")
        
        final_percentage = (remaining_battery_kwh / total_capacity) * 100
        metrics_col3.metric("Ending Battery State", f"{max(0.0, final_percentage):.1f} %")
        
        st.markdown("---")
        
        if remaining_battery_kwh <= 0:
            st.error(f"🚨 **CRITICAL RANGE ALERT:** Your EV will run completely out of charge! You will stall approximately {abs(remaining_battery_kwh):.2f} kWh short of your destination.")
        elif remaining_battery_kwh < safety_limit:
            st.warning(f"⚠️ **RANGE ANXIETY WARNING:** You will arrive, but your final battery state will drop to {final_percentage:.1f}%, dipping below your 10% safety margin.")
        else:
            st.success(f"✅ **OPTIMAL ROUTE PROFILE:** You will arrive safely at your destination with {final_percentage:.1f}% charge remaining.")