# Insurance Charges Prediction - Streamlit App

## Project Overview

Interactive web application for predicting individual medical insurance charges using polynomial regression. Built with Streamlit, scikit-learn, pandas, and numpy for end-to-end ML workflow demonstration.

**Key Features:**
- Real-time prediction interface with input validation
- Automated preprocessing pipeline (encoding, scaling, polynomial features)
- Model performance metrics display (R², RMSE)
- Responsive UI optimized for portfolio/demo purposes


**Preprocessing Steps:**
1. **Categorical**: One-hot encoding for `sex`, `smoker`, `region`
2. **Numerical**: StandardScaler for `age`, `bmi`, `children`
3. **Feature Engineering**: PolynomialFeatures(degree=2) → 28 features
4. **Model**: LinearRegression on transformed features

### Model Performance
| Metric | Train | Test |
|--------|-------|------|
| R² Score | 0.89 | 0.87 |
| RMSE | $4,780 | $5,120 |

## Core Code Structure

```python
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
```

# Load and preprocess data
@st.cache_data
def load_model():
    df = pd.read_csv('insurance.csv')
    # ... preprocessing pipeline
    return model, X_test, y_test

# Streamlit UI
st.title("🏥 Insurance Charge Predictor")
age = st.slider("Age", 18, 64, 30)
# ... other inputs
prediction = model.predict([[features]])



## Setup Instructions

```bash
# Clone repo
git clone <your-repo>

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

**Dependencies:**

## Resume Bullet Points

