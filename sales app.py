
---

## **📜 app.py**
```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Page config
st.set_page_config(page_title="Sales Prediction App", layout="centered")

# Title
st.title("📊 Sales Prediction using Machine Learning")
st.write("Predict product sales based on advertising spend on TV, Radio, and Newspaper.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/advertising.csv")

df = load_data()

# EDA - Correlation heatmap
st.subheader("📈 Data Overview")
st.write(df.head())

corr = df.corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Train model
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
st.write("**R² Score:**", round(r2_score(y_test, y_pred), 3))
st.write("**RMSE:**", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))

# Prediction input
st.subheader("🛠 Try It Yourself")
tv = st.slider("TV Advertising Budget ($)", 0, 300, 100)
radio = st.slider("Radio Advertising Budget ($)", 0, 50, 25)
news = st.slider("Newspaper Advertising Budget ($)", 0, 120, 20)

# Predict
pred = model.predict([[tv, radio, news]])[0]
st.success(f"Predicted Sales: {pred:.2f} (in thousands of units)")
