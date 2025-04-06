# NYC Taxi Fare Prediction 🚖 | April 2024 Data

This project predicts NYC green taxi fare amounts using April 2024 data and machine learning. A user-friendly Streamlit web application is provided to test predictions by inputting trip details like distance, pickup hour, and more.

---

## 🔍 Project Overview

This project aims to:
- Analyze real-world NYC Green Taxi data (April 2024)
- Build a regression model to predict fare amount
- Deploy a live interactive app using Streamlit

---

## 📊 Dataset

- Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- File: `green_tripdata_2024-04.parquet`
- Format: Parquet
- Size: ~140MB+
- Features used:
  - `trip_distance`
  - `passenger_count`
  - `pickup_hour`
  - `pickup_dayofweek`

---

## ⚙️ Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn (Linear Regression)
- Matplotlib & Seaborn (EDA)
- Streamlit (Web App)

---

## 🚀 How to Run Locally

1. **Clone this repo**:
   ```bash
   git clone https://github.com/your-username/nyc-taxi-fare-prediction.git
   cd nyc-taxi-fare-prediction
