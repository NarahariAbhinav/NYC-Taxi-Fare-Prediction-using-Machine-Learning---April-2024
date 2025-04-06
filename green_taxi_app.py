import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Conditionally import seaborn
try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False
    st.warning("Seaborn is not installed. Some visualizations will be simplified.")

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
        
        # Check if coordinates exist in the dataset
        has_pickup_coords = all(col in filtered_df.columns for col in ['pickup_longitude', 'pickup_latitude'])
        has_dropoff_coords = all(col in filtered_df.columns for col in ['dropoff_longitude', 'dropoff_latitude'])
        
        # If location IDs exist but no coordinates, inform the user
        if not has_pickup_coords and 'PULocationID' in filtered_df.columns:
            st.info("Your dataset contains location IDs but not direct coordinates. Consider using a lookup table to map location IDs to coordinates.")
        
        # Only show map options if coordinates are available
        if has_pickup_coords or has_dropoff_coords:
            map_options = []
            if has_pickup_coords:
                map_options.append("Pickup Locations")
            if has_dropoff_coords:
                map_options.append("Dropoff Locations")
            if has_pickup_coords and has_dropoff_coords:
                map_options.append("Pickup-Dropoff Flow")
                
            map_type = st.radio("Map Type", map_options, horizontal=True)
            
            # Limit data points for better performance
            map_sample = filtered_df.sample(min(len(filtered_df), 10000))
            
            if map_type == "Pickup Locations" and has_pickup_coords:
                st.write("### Pickup Location Heatmap")
                
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
            
            elif map_type == "Dropoff Locations" and has_dropoff_coords:
                st.write("### Dropoff Location Heatmap")
                
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
            
            elif map_type == "Pickup-Dropoff Flow" and has_pickup_coords and has_dropoff_coords:
                st.write("### Pickup to Dropoff Flow")
                
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
            st.warning("No coordinate columns found in the dataset. Map visualizations require longitude and latitude data.")
            
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
            if seaborn_available:
                sns.histplot(filtered_df['total_amount'], bins=50, kde=True, ax=ax1)
            else:
                ax1.hist(filtered_df['total_amount'], bins=50)
            ax1.set_xlabel('Total Amount ($)')
            ax1.set_ylabel('Frequency')
            st.pyplot(fig1)
            
        with col2:
            st.markdown("### 🔹 Average Fare by Weekday")
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            avg_fare = filtered_df.groupby('weekday')['total_amount'].mean()
            # Reindex if all weekdays are present
            if set(weekday_order).issubset(set(filtered_df['weekday'].unique())):
                avg_fare = avg_fare.reindex(weekday_order)
            st.bar_chart(avg_fare)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 🔸 Trip Duration vs Fare")
            fig2, ax2 = plt.subplots()
            sample_df = filtered_df.sample(min(len(filtered_df), 5000))
            if seaborn_available:
                sns.scatterplot(data=sample_df, x='trip_duration', y='fare_amount', alpha=0.5, ax=ax2)
            else:
                ax2.scatter(sample_df['trip_duration'], sample_df['fare_amount'], alpha=0.5)
            ax2.set_xlabel('Trip Duration (minutes)')
            ax2.set_ylabel('Fare Amount ($)')
            st.pyplot(fig2)
        
        with col4:
            st.markdown("### 🔹 Passenger Count vs Total Amount")
            fig3, ax3 = plt.subplots()
            if seaborn_available:
                sns.boxplot(data=filtered_df, x='passenger_count', y='total_amount', ax=ax3)
            else:
                # Simple alternative using matplotlib
                passenger_counts = sorted(filtered_df['passenger_count'].unique())
                boxplot_data = [filtered_df[filtered_df['passenger_count'] == count]['total_amount'] for count in passenger_counts]
                ax3.boxplot(boxplot_data)
                ax3.set_xticklabels(passenger_counts)
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
        
        # Add legend handling
        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax5.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc='upper right')
        st.pyplot(fig4)
        
    with tab4:
        # Features for model
        st.subheader("🧠 Train Model & Predict Fare")
        
        features = ['trip_distance', 'trip_duration', 'passenger_count']
        # Add optional features if they exist in the dataset
        optional_features = ['fare_amount', 'extra', 'mta_tax', 'tip_amount',
                           'tolls_amount', 'congestion_surcharge']
        for feature in optional_features:
            if feature in filtered_df.columns:
                features.append(feature)
        
        # Ensure all selected features are in the dataframe
        features = [f for f in features if f in filtered_df.columns]
        
        # Check if we have enough features to build a model
        if len(features) < 2:
            st.warning("Not enough features available to build a prediction model. Please ensure your dataset contains at least trip_distance, trip_duration, or passenger_count.")
        else:
            # Display selected features
            st.write(f"Using features: {', '.join(features)}")
            
            # Prepare data for modeling
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
