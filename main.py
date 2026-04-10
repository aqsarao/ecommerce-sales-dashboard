import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("E-Commerce Sales Dashboard")

# Load data (you can also use uploader later)
df = pd.read_csv("data/ecommerce.csv")

# Data preprocessing
df['date'] = pd.to_datetime(df['date'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

# Feature engineering
df['revenue'] = df['price'] * df['quantity']
df['month'] = df['date'].dt.to_period('M')

# Sidebar filter
st.sidebar.header("Filter Data")
selected_city = st.sidebar.selectbox("Select City", df['city'].unique())

filtered_df = df[df['city'] == selected_city]

# Show data
st.subheader("Filtered Data")
st.write(filtered_df.head())

# Total revenue
st.metric("Total Revenue", int(filtered_df['revenue'].sum()))

# Revenue by product
st.subheader("Revenue by Product")
product_revenue = filtered_df.groupby('product')['revenue'].sum().sort_values(ascending=False)

fig, ax = plt.subplots()
sns.barplot(x=product_revenue.index, y=product_revenue.values, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Monthly trend
st.subheader("Monthly Revenue Trend")
monthly_revenue = filtered_df.groupby('month')['revenue'].sum()

fig2, ax2 = plt.subplots()
monthly_revenue.plot(marker='o', ax=ax2)
st.pyplot(fig2)

# Insights
st.subheader("Key Insights")

top_product = filtered_df.groupby('product')['revenue'].sum().idxmax()
top_city = filtered_df.groupby('city')['revenue'].sum().idxmax()

st.write(f" Top Product: {top_product}")
st.write(f" Top City: {top_city}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.subheader(" Sales Prediction (ML Model)")

# Features (X) and Target (y)
X = df[['quantity']]   # you can later add more features
y = df['revenue']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# User input
input_quantity = st.number_input("Enter Quantity to Predict Revenue", min_value=1, value=1)

# Prediction
prediction = model.predict([[input_quantity]])

st.write(f" Predicted Revenue: {int(prediction[0])}")

from sklearn.metrics import r2_score

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

st.write(f" Model Accuracy (R² Score): {round(score, 2)}")