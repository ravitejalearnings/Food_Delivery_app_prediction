import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder


file_path1 = "model.pkl"
with open(file_path1,'rb') as f:
    model = pickle.load(f)

file_path2 = "scaler.pkl"
with open(file_path2,'rb') as f:
    scaler = pickle.load(f)

st.title('Delivery Time Prediction')

# User input for each feature
delivery_person_id = st.text_input('Delivery Person ID', 'BANGRES19DEL01')
age = st.number_input('Delivery Person Age', min_value=18, max_value=65, value=30)
ratings = st.number_input('Delivery Person Ratings', min_value=1.0, max_value=5.0, value=4.5)
weather = st.selectbox('Weather Conditions', ['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog', 'Windy'])
traffic = st.selectbox('Road Traffic Density', ['Low', 'Medium', 'High', 'Jam'])
vehicle_condition = st.number_input('Vehicle Condition', min_value=0, max_value=10, value=7)
order_type = st.selectbox('Type of Order', ['Snack', 'Meal', 'Drinks', 'Buffet'])
vehicle_type = st.selectbox('Type of Vehicle', ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
multiple_deliveries = st.number_input('Multiple Deliveries', min_value=0, max_value=5, value=0)
festival = st.selectbox('Festival', ['No', 'Yes'])
city = st.selectbox('City', ['Urban', 'Semi-Urban', 'Metropolitan'])
distance = st.number_input('Distance', min_value=1, max_value=30, value=1)
week = st.selectbox('week(y/n', ['weekend','weekday'])

road_traffic_density_mapping = {
    'High': 4,
    'Jam': 3,
    'Medium': 2,
    'Low': 1
}
city_mapping = {
    'Urban':2,
    'Semi-Urban':1,
    'Metropolitian':3
}
week_mapping = {
    'weekend': 1,
    'weekday': 0
}
festival_mapping = {
    'No': 0,
    'Yes': 1
}


if st.button("Get ETA for Delivery!"):

    input_df = pd.DataFrame({
        'Delivery_person_ID': [delivery_person_id],
        'Delivery_person_Age': [age],
        'Delivery_person_Ratings': [ratings],
        'Weather_conditions': [weather],
        'Road_traffic_density': [traffic],
        'Vehicle_condition': [vehicle_condition],
        'multiple_deliveries': [multiple_deliveries],
        'Festival': [festival],
        'City': [city],
        'city_code': [delivery_person_id],
        'week': [week],
        'Distance': [distance],
        'Type_of_order': [order_type],
        'Type_of_vehicle': [vehicle_type]
    })

    input_df['week(y/n'] = input_df['week'].map(week_mapping)
    input_df['city_code'] = input_df['Delivery_person_ID'].str.split('RES', expand=True)[0]
    input_df['Delivery_person_ID'] = input_df['Delivery_person_ID'].str.split('RES', expand=True)[1]
    input_df['Road_traffic_density'] = input_df['Road_traffic_density'].map(road_traffic_density_mapping)
    input_df['City'] = input_df['City'].map(city_mapping)
    input_df['Festival'] = input_df['Festival'].map(festival_mapping)


    input_df['weatherconditions_fog'] = input_df['Weather_conditions'].apply(lambda x: 1 if x == 'fog' else 0)
    input_df['weatherconditions_sandstorms'] = input_df['Weather_conditions'].apply(lambda x: 1 if x == 'sandstorms' else 0)
    input_df['weatherconditions_stormy'] = input_df['Weather_conditions'].apply(lambda x: 1 if x == 'stormy' else 0)
    input_df['weatherconditions_sunny'] = input_df['Weather_conditions'].apply(lambda x: 1 if x == 'sunny' else 0)
    input_df['weatherconditions_windy'] = input_df['Weather_conditions'].apply(lambda x: 1 if x == 'windy' else 0)

    input_df['type_of_order_drinks'] = input_df['Type_of_order'].apply(lambda x: 1 if x == 'drinks' else 0)
    input_df['type_of_order_meal'] = input_df['Type_of_order'].apply(lambda x: 1 if x == 'meal' else 0)
    input_df['type_of_order_snack'] = input_df['Type_of_order'].apply(lambda x: 1 if x == 'snack' else 0)

    input_df['type_of_vehicle_electric_scooter'] = input_df['Type_of_vehicle'].apply(lambda x: 1 if x == 'electric_scooter' else 0)
    input_df['type_of_vehicle_motorcycle'] = input_df['Type_of_vehicle'].apply(lambda x: 1 if x == 'motorcycle' else 0)
    input_df['type_of_vehicle_scooter'] = input_df['Type_of_vehicle'].apply(lambda x: 1 if x == 'scooter' else 0)

    input_df.columns = input_df.columns.str.lower()
    cols_to_drop = ['weather_conditions','type_of_vehicle','type_of_order','week']

    input_data = input_df.drop(columns=cols_to_drop, axis=1)

    order_based_on_train = ['delivery_person_id', 'delivery_person_age', 'delivery_person_ratings',
       'road_traffic_density', 'vehicle_condition', 'multiple_deliveries',
       'festival', 'city', 'city_code', 'week(y/n', 'distance',
       'weatherconditions_fog', 'weatherconditions_sandstorms',
       'weatherconditions_stormy', 'weatherconditions_sunny',
       'weatherconditions_windy', 'type_of_order_drinks', 'type_of_order_meal',
       'type_of_order_snack', 'type_of_vehicle_electric_scooter',
       'type_of_vehicle_motorcycle', 'type_of_vehicle_scooter']

    input_data = input_df[order_based_on_train]

    cols_to_label_encode = input_data.select_dtypes(include='object').columns
    encoder = LabelEncoder()
    for i in cols_to_label_encode:
        encoder.fit(input_data[i])
        input_data[i] = encoder.transform(input_data[i])

    cols_to_scale = ['delivery_person_id', 'delivery_person_age', 'delivery_person_ratings',
       'road_traffic_density', 'vehicle_condition', 'multiple_deliveries',
       'festival', 'city', 'city_code', 'week(y/n', 'distance']

    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    prediction = model.predict(input_data)

    st.write(f'Your Food will arrive in minutes 😋😋 {prediction[0].astype('int')} mins')
