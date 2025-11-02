# bookshop_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Bikrampur Boighar Dashboard")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    # 3 Years Sales
    df1 = pd.read_excel("Sell- 2023.xlsx")
    df1['Year'] = 2023
    df2 = pd.read_excel("Sell - 2024.xlsx")
    df2['Year'] = 2024
    df3 = pd.read_excel("Sell - 2025.xlsx")
    df3['Year'] = 2025
    df_sales = pd.concat([df1, df2, df3], ignore_index=True)
    
    # All Book List
    df_books = pd.read_excel("All Book List.xlsx")
    
    # Clean Names
    df_sales['Name of Books'] = df_sales['Name of Books'].str.strip().str.lower()
    df_books['Name of Books'] = df_books['Name of Books'].str.strip().str.lower()
    
    # Merge
    df = df_sales.merge(df_books, on='Name of Books', how='left')
    
    # Fill missing numeric
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Repeated Customer
    df = df.dropna(subset=['Repeated Customer?'])
    df['Repeated_Customer'] = df['Repeated Customer?'].str.strip().str.upper().map({'YES':1,'NO':0})
    
    # Profit
    df['Profit'] = df['Sell'] - df['Cost']
    
    # Age Group
    age_bins = [0,20,30,40,50,60,100]
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins)
    
    # Encode categorical
    cat_cols = ['Gender','Delivery Method','Category Of Books',"Book's Genre",'Publication']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Month
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Month'] = df['Order Date'].dt.month
    return df

df = load_data()

# ----------------- SIDEBAR -----------------
st.sidebar.title("Filters")
years = sorted(df['Year'].unique())
year_filter = st.sidebar.multiselect("Select Year", years, default=years)
categories = df['Category Of Books'].unique()
category_filter = st.sidebar.multiselect("Select Category", categories, default=categories)

df_filtered = df[(df['Year'].isin(year_filter)) & (df['Category Of Books'].isin(category_filter))]

# ----------------- METRICS -----------------
st.title("Bikrampur Boighar Dashboard")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sell", int(df_filtered['Sell'].sum()))
col2.metric("Total Profit", int(df_filtered['Profit'].sum()))
col3.metric("Total Unique Books Sold", df_filtered['Name of Books'].nunique())
col4.metric("Repeated Customers", int(df_filtered['Repeated_Customer'].sum()))

# ----------------- SALES TREND -----------------
st.subheader("Monthly Sales Trend")
monthly_sales = df_filtered.groupby('Month')['Sell'].sum()
fig, ax = plt.subplots()
monthly_sales.plot(kind='line', marker='o', ax=ax)
ax.set_ylabel("Sell")
ax.set_xlabel("Month")
st.pyplot(fig)

# ----------------- CATEGORY & GENRE SALES -----------------
st.subheader("Category-wise Sales")
category_sales = df_filtered.groupby('Category Of Books')['Sell'].sum()
fig, ax = plt.subplots()
category_sales.plot(kind='bar', color='lightgreen', ax=ax)
st.pyplot(fig)

st.subheader("Genre-wise Sales")
genre_sales = df_filtered.groupby("Book's Genre")['Sell'].sum()
fig, ax = plt.subplots()
genre_sales.plot(kind='bar', color='skyblue', ax=ax)
st.pyplot(fig)

# ----------------- GENDER RETENTION -----------------
st.subheader("Gender-wise Repeated Customer (%)")
gender_ret = df_filtered.groupby('Gender')['Repeated_Customer'].mean()*100
fig, ax = plt.subplots()
gender_ret.plot(kind='bar', color=['pink','lightblue'], ax=ax)
st.pyplot(fig)

# ----------------- DELIVERY METHOD -----------------
st.subheader("Delivery Method Usage (%)")
delivery_counts = df_filtered['Delivery Method'].value_counts(normalize=True)*100
fig, ax = plt.subplots()
delivery_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
st.pyplot(fig)

# ----------------- CUSTOMER SEGMENTATION -----------------
st.subheader("Customer Segmentation (KMeans Clusters)")
cluster_features = ['Age','Gender','Unit','Profit']
X_cluster = df_filtered[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_ret = df_filtered.groupby('Cluster')['Repeated_Customer'].mean()*100
fig, ax = plt.subplots()
sns.heatmap(cluster_ret.to_frame().T, annot=True, fmt=".2f", cmap='YlGnBu', ax=ax)
st.pyplot(fig)

# ----------------- ML PREDICTION WIDGET -----------------
st.subheader("Predict Sell & Repeated Customer")

age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", options=[0,1])  # 0=Female,1=Male
unit = st.number_input("Units", min_value=1, max_value=100, value=1)
profit = st.number_input("Profit per Book", min_value=0, value=10)

# Regression Model
X_train, X_test, y_train, y_test = train_test_split(df_filtered[['Age','Gender','Unit','Profit']], df_filtered['Sell'], test_size=0.2, random_state=42)
model_reg = LinearRegression().fit(X_train, y_train)

# Classification Model
Xc_train, Xc_test, yc_train, yc_test = train_test_split(df_filtered[['Age','Gender','Unit','Profit']], df_filtered['Repeated_Customer'], test_size=0.2, random_state=42)
model_clf = LogisticRegression(max_iter=1000).fit(Xc_train, yc_train)

if st.button("Predict"):
    X_input = pd.DataFrame([[age, gender, unit, profit]], columns=['Age','Gender','Unit','Profit'])
    sell_pred = model_reg.predict(X_input)[0]
    retain_pred = model_clf.predict(X_input)[0]
    st.success(f"Predicted Sell: {sell_pred:.0f}")
    st.success(f"Repeated Customer Likely: {'Yes' if retain_pred==1 else 'No'}")
