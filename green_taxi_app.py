import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Page config
st.set_page_config(page_title="🚖 NYC Green Taxi - April 2024", layout="wide")
st.title("🚖 NYC Green Taxi Dashboard – April 2024")

# Upload parquet
file = st.file_uploader("📂 Upload April 2024 Parquet File", type=["parquet"])

if file:
    df = pd.read_parquet(file)

    # Preprocessing
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['weekday'] = df['lpep_pickup_datetime'].dt.day_name()
    df['hourofday'] = df['lpep_pickup_datetime'].dt.hour
    df['date'] = df['lpep_pickup_datetime'].dt.date

    df.fillna(df.median(numeric_only=True), inplace=True)

    # Add sidebar filters
    st.sidebar.header("🔍 Filter Options")
    
    # Date filter
    date_options = ['All'] + sorted(df['date'].unique().astype(str).tolist())
    selected_date = st.sidebar.selectbox("📅 Select Date", date_options)
    
    # Weekday filter
    weekday_options = ['All'] + sorted(df['weekday'].unique().tolist())
    selected_day = st.sidebar.selectbox("📅 Select Weekday", weekday_options)
    
    # Hour range filter
    hour_range = st.sidebar.slider("🕒 Hour Range", 0, 23, (0, 23))
    
    # Apply filters
    filtered_df = df.copy()
    if selected_date != 'All':
        filtered_df = filtered_df[filtered_df['date'].astype(str) == selected_date]
    if selected_day != 'All':
        filtered_df = filtered_df[filtered_df['weekday'] == selected_day]
    filtered_df = filtered_df[(filtered_df['hourofday'] >= hour_range[0]) & 
                              (filtered_df['hourofday'] <= hour_range[1])]

    # KPI metrics
    st.subheader("📊 Key Metrics")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Total Trips", f"{len(filtered_df):,}")
    with metrics_col2:
        st.metric("Avg Fare", f"${filtered_df['fare_amount'].mean():.2f}")
    with metrics_col3:
        st.metric("Avg Trip Distance", f"{filtered_df['trip_distance'].mean():.2f} mi")
    with metrics_col4:
        st.metric("Avg Tip", f"${filtered_df['tip_amount'].mean():.2f}")

    # Dataset preview tab
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🗺️ Maps", "📈 Analytics", "🧠 Predictions"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(filtered_df.head(100), use_container_width=True)
        
        # Data summary
        if st.checkbox("Show Data Summary"):
            st.write("### Data Summary")
            st.write(filtered_df.describe())
    
    with tab2:
        st.subheader("🗺️ Map Visualizations")
        map_type = st.radio("Map Type", ["Pickup Locations", "Dropoff Locations", "Pickup-Dropoff Flow"], horizontal=True)
        
        # Limit data points for better performance
        map_sample = filtered_df.sample(min(len(filtered_df), 10000))
        
        if map_type == "Pickup Locations":
            st.write("### Pickup Location Heatmap")
            
            # Check if pickup coordinates are available
            if 'pickup_longitude' in map_sample.columns and 'pickup_latitude' in map_sample.columns:
                # Filter valid coordinates
                valid_pickups = map_sample[(map_sample['pickup_longitude'] != 0) & 
                                           (map_sample['pickup_latitude'] != 0) &
                                           (map_sample['pickup_longitude'] > -180) & 
                                           (map_sample['pickup_longitude'] < 180) &
                                           (map_sample['pickup_latitude'] > -90) & 
                                           (map_sample['pickup_latitude'] < 90)]
                
                if len(valid_pickups) > 0:
                    st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v10',
                        initial_view_state=pdk.ViewState(
                            latitude=40.7128,
                            longitude=-74.0060,
                            zoom=10,
                            pitch=0,
                        ),
                        layers=[
                            pdk.Layer(
                                'HeatmapLayer',
                                data=valid_pickups,
                                get_position=['pickup_longitude', 'pickup_latitude'],
                                get_weight='fare_amount',
                                radiusPixels=50,
                                opacity=0.7,
                            )
                        ],
                    ))
                else:
                    st.warning("No valid pickup coordinates found in the dataset.")
            else:
                st.warning("Pickup coordinates not found in the dataset. Please make sure your data includes pickup_longitude and pickup_latitude columns.")
        
        elif map_type == "Dropoff Locations":
            st.write("### Dropoff Location Heatmap")
            
            # Check if dropoff coordinates are available
            if 'dropoff_longitude' in map_sample.columns and 'dropoff_latitude' in map_sample.columns:
                # Filter valid coordinates
                valid_dropoffs = map_sample[(map_sample['dropoff_longitude'] != 0) & 
                                            (map_sample['dropoff_latitude'] != 0) &
                                            (map_sample['dropoff_longitude'] > -180) & 
                                            (map_sample['dropoff_longitude'] < 180) &
                                            (map_sample['dropoff_latitude'] > -90) & 
                                            (map_sample['dropoff_latitude'] < 90)]
                
                if len(valid_dropoffs) > 0:
                    st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v10',
                        initial_view_state=pdk.ViewState(
                            latitude=40.7128,
                            longitude=-74.0060,
                            zoom=10,
                            pitch=0,
                        ),
                        layers=[
                            pdk.Layer(
                                'HeatmapLayer',
                                data=valid_dropoffs,
                                get_position=['dropoff_longitude', 'dropoff_latitude'],
                                get_weight='fare_amount',
                                radiusPixels=50,
                                opacity=0.7,
                            )
                        ],
                    ))
                else:
                    st.warning("No valid dropoff coordinates found in the dataset.")
            else:
                st.warning("Dropoff coordinates not found in the dataset. Please make sure your data includes dropoff_longitude and dropoff_latitude columns.")
        
        else:  # Pickup-Dropoff Flow
            st.write("### Pickup to Dropoff Flow")
            
            # Check if pickup and dropoff coordinates are available
            if all(col in map_sample.columns for col in ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']):
                # Create flow data
                flow_data = map_sample[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']].copy()
                
                # Filter valid coordinates
                valid_flows = flow_data[(flow_data['pickup_longitude'] != 0) & 
                                        (flow_data['pickup_latitude'] != 0) &
                                        (flow_data['dropoff_longitude'] != 0) & 
                                        (flow_data['dropoff_latitude'] != 0) &
                                        (flow_data['pickup_longitude'] > -180) & 
                                        (flow_data['pickup_longitude'] < 180) &
                                        (flow_data['pickup_latitude'] > -90) & 
                                        (flow_data['pickup_latitude'] < 90) &
                                        (flow_data['dropoff_longitude'] > -180) & 
                                        (flow_data['dropoff_longitude'] < 180) &
                                        (flow_data['dropoff_latitude'] > -90) & 
                                        (flow_data['dropoff_latitude'] < 90)]
                
                # Limit to fewer trips for performance
                valid_flows = valid_flows.sample(min(len(valid_flows), 2000))
                
                if len(valid_flows) > 0:
                    # Create arc layer data
                    arc_data = []
                    for _, row in valid_flows.iterrows():
                        arc_data.append({
                            'sourcePosition': [float(row['pickup_longitude']), float(row['pickup_latitude'])],
                            'targetPosition': [float(row['dropoff_longitude']), float(row['dropoff_latitude'])],
                            'fare': float(row['fare_amount'])
                        })
                    
                    st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/dark-v10',
                        initial_view_state=pdk.ViewState(
                            latitude=40.7128,
                            longitude=-74.0060,
                            zoom=10,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                'ArcLayer',
                                data=arc_data,
                                get_source_position='sourcePosition',
                                get_target_position='targetPosition',
                                get_width=1,
                                get_source_color=[0, 255, 0, 150],
                                get_target_color=[255, 0, 0, 150],
                                get_height=0.5,
                                pickable=True,
                            )
                        ],
                        tooltip={"text": "Fare: ${fare}"}
                    ))
                else:
                    st.warning("No valid coordinates for flows found in the dataset.")
            else:
                st.warning("Required coordinate columns not found in the dataset.")
        
        # Geographic analysis
        st.write("### 📍 Popular Zones Analysis")
        if 'PULocationID' in filtered_df.columns:
            pickup_zones = filtered_df['PULocationID'].value_counts().head(10)
            st.bar_chart(pickup_zones)
        
    with tab3:
        st.subheader("📈 Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔸 Total Amount Distribution")
            fig1, ax1 = plt.subplots()
            sns.histplot(filtered_df['total_amount'], bins=50, kde=True, ax=ax1)
            ax1.set_xlabel('Total Amount ($)')
            ax1.set_ylabel('Frequency')
            st.pyplot(fig1)
            
        with col2:
            st.markdown("### 🔹 Average Fare by Weekday")
            avg_fare = filtered_df.groupby('weekday')['total_amount'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            st.bar_chart(avg_fare)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 🔸 Trip Duration vs Fare")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=filtered_df.sample(min(len(filtered_df), 5000)), x='trip_duration', y='fare_amount', alpha=0.5, ax=ax2)
            ax2.set_xlabel('Trip Duration (minutes)')
            ax2.set_ylabel('Fare Amount ($)')
            st.pyplot(fig2)
        
        with col4:
            st.markdown("### 🔹 Passenger Count vs Total Amount")
            fig3, ax3 = plt.subplots()
            sns.boxplot(data=filtered_df, x='passenger_count', y='total_amount', ax=ax3)
            ax3.set_xlabel('Passenger Count')
            ax3.set_ylabel('Total Amount ($)')
            st.pyplot(fig3)
        
        # Time-based analysis
        st.markdown("### 🕒 Trip Patterns by Hour")
        hourly_trips = filtered_df.groupby('hourofday').size()
        hourly_fares = filtered_df.groupby('hourofday')['fare_amount'].mean()
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.bar(hourly_trips.index, hourly_trips, alpha=0.7, label='Trip Count')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Number of Trips')
        
        ax5 = ax4.twinx()
        ax5.plot(hourly_fares.index, hourly_fares, color='red', marker='o', linestyle='-', linewidth=2, label='Avg Fare')
        ax5.set_ylabel('Average Fare ($)')
        
        fig4.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax4.transAxes)
        st.pyplot(fig4)
        
    with tab4:
        # Features for model
        st.subheader("🧠 Train Model & Predict Fare")
        
        features = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                   'tolls_amount', 'congestion_surcharge', 'trip_duration', 'passenger_count']
        features = [f for f in features if f in filtered_df.columns]
        
        X = filtered_df[features]
        y = filtered_df['total_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            model_type = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                n_estimators = st.slider("Number of trees", 10, 200, 100)
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            
            start_training = st.button("Train Model")
        
        with model_col2:
            if start_training:
                with st.spinner('Training model...'):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    st.success(f"✅ Model trained successfully!")
                    st.metric("R² Score", f"{r2:.3f}")
                    st.metric("Mean Absolute Error", f"${mae:.2f}")
                    
                    if model_type == "Random Forest":
                        # Feature importance
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        
                        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                        ax_imp.barh(range(len(indices)), importances[indices], align='center')
                        ax_imp.set_yticks(range(len(indices)))
                        ax_imp.set_yticklabels([features[i] for i in indices])
                        ax_imp.set_xlabel('Feature Importance')
                        ax_imp.set_title('Feature Importance for Fare Prediction')
                        st.pyplot(fig_imp)
        
        # Prediction interface
        st.subheader("🔮 Predict Fare from Your Input")
        
        input_cols = st.columns(3)
        input_data = {}
        
        for i, feature in enumerate(features):
            with input_cols[i % 3]:
                min_val = float(filtered_df[feature].min())
                max_val = float(filtered_df[feature].max())
                mean_val = float(filtered_df[feature].mean())
                input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}",
                                                     min_value=min_val,
                                                     max_value=max_val,
                                                     value=mean_val,
                                                     format="%.2f")
        
        # Make prediction
        predict_col1, predict_col2 = st.columns([1, 2])
        
        with predict_col1:
            predict_button = st.button("Predict Fare")
        
        with predict_col2:
            if predict_button and 'model' in locals():
                input_df = pd.DataFrame([input_data])
                predicted_fare = model.predict(input_df)[0]
                st.success(f"💵 Predicted Total Fare Amount: ${predicted_fare:.2f}")
            elif predict_button:
                st.warning("Please train the model first")