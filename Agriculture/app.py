import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@st.cache_data
def load_data():
    df = pd.read_csv("FAOSTAT.csv")

    df.columns = df.columns.str.strip()
    df['Element'] = df['Element'].str.strip()
    df['Item'] = df['Item'].str.strip()  

    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    df = df[df['Element'].isin(['Area harvested', 'Yield', 'Production'])]

    df = df.pivot_table(index=['Area', 'Item', 'Year'], 
                        columns='Element', 
                        values='Value', 
                        aggfunc='sum').reset_index()

    df.rename(columns={'Area harvested': 'Area_Harvested', 
                       'Yield': 'Yield_kg_ha', 
                       'Production': 'Production_t'}, inplace=True)

    numeric_cols = ['Area_Harvested', 'Yield_kg_ha', 'Production_t']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    df['Year'] = df['Year'].astype(int)

    return df

df = load_data()

if df.empty:
    st.error("Error: Dataset is empty after preprocessing. Please check the data file.")
    st.stop()

features = ['Area_Harvested', 'Yield_kg_ha', 'Year']
target = 'Production_t'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

def main():    
    st.title("🌾 Crop Production Prediction App")
    st.write("This app predicts crop production based on area harvested, yield, and year.")
    
    st.sidebar.header("User Input")
    area = st.sidebar.selectbox("Select Region", df['Area'].unique())
    crop = st.sidebar.selectbox("Select Crop", df['Item'].unique())
    year = st.sidebar.slider("Select Year", int(df['Year'].min()), int(df['Year'].max()), step=1)
    area_harvested = st.sidebar.number_input("Enter Area Harvested (ha)", min_value=0.0, value=1000.0, step=100.0)
    yield_per_ha = st.sidebar.number_input("Enter Yield (kg/ha)", min_value=0.0, value=1500.0, step=100.0)
    
    if st.sidebar.button("Predict Production"):
        input_data = np.array([[area_harvested, yield_per_ha, year]])
        prediction = model.predict(input_data)
        st.sidebar.success(f"🌟 Predicted Production: {prediction[0]:,.2f} tons")
        if prediction[0] > 5000:
            st.sidebar.write("✅ This region is predicted to have a high crop production. Consider focusing on maximizing yield!")
        elif prediction[0] < 500:
            st.sidebar.write("⚠️ This region is predicted to have low crop production. It may require intervention, such as improved farming techniques or crop diversification.")
        else:
            st.sidebar.write("📈 This region has moderate crop production. Monitoring and resource optimization can improve outcomes.")
    
    st.subheader("📊 Model Performance")
    st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
    st.write(f"**R² Score:** {r2:.4f}")
    
    st.subheader("📈 Crop Production Trends")
    filtered_data = df[(df['Item'] == crop) & (df['Area'] == area)]
    fig = px.line(filtered_data, x='Year', y='Production_t', title=f"Crop Production of {crop} in {area}")
    st.plotly_chart(fig)
    
    st.subheader("🌍 Yearly Trends Across All Crops")
    fig2 = px.line(df, x='Year', y='Production_t', color='Item', title="Yearly Trends of Crop Production")
    st.plotly_chart(fig2)

if __name__ == "__main__":
    main()
