import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title='Insurance Charges Predictor', page_icon='💳', layout='wide')

@st.cache_data
def load_data():
    return pd.read_csv('insurance.csv')

@st.cache_resource
def train_model(df):
    df = df.copy()
    df['charges_log'] = np.log1p(df['charges'])
    X = df.drop(columns=['charges', 'charges_log'])
    y = df['charges_log']

    X_enc = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)
    feature_columns = X_enc.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

    num_cols = ['age', 'bmi', 'children']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)

    metrics = {
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'test_rmse_log': np.sqrt(mean_squared_error(y_test, test_pred)),
        'sample_size': len(df)
    }
    return model, scaler, poly, feature_columns, metrics


def prepare_input(age, sex, bmi, children, smoker, region, feature_columns, scaler):
    input_df = pd.DataFrame([{ 
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])
    input_enc = pd.get_dummies(input_df, columns=['sex', 'smoker', 'region'], drop_first=True)
    input_enc = input_enc.reindex(columns=feature_columns, fill_value=0)
    input_enc[['age', 'bmi', 'children']] = scaler.transform(input_enc[['age', 'bmi', 'children']])
    return input_enc

st.markdown("""
    <style>
    .main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;}
    .sub-text {color: #6b7280; margin-bottom: 1.2rem;}
    .metric-card {padding: 1rem; border-radius: 14px; background: #f5f7fb; border: 1px solid #e5e7eb;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>💳 Insurance Charges Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Interactive Streamlit UI for predicting medical insurance charges using a polynomial regression pipeline built with preprocessing, encoding, scaling, and model evaluation.</div>", unsafe_allow_html=True)

try:
    df = load_data()
    model, scaler, poly, feature_columns, metrics = train_model(df)
except Exception as e:
    st.error("Could not load insurance.csv. Place the dataset in the same folder as app.py before running the app.")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader('Applicant details')
    age = st.slider('Age', 18, 64, 30)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.slider('BMI', 15.0, 55.0, 28.5, 0.1)
    children = st.slider('Children', 0, 5, 0)
    smoker = st.selectbox('Smoker', ['no', 'yes'])
    region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

    if st.button('Predict charges', type='primary', use_container_width=True):
        X_input = prepare_input(age, sex, bmi, children, smoker, region, feature_columns, scaler)
        X_poly = poly.transform(X_input)
        pred_log = model.predict(X_poly)[0]
        pred = np.expm1(pred_log)
        st.success(f'Estimated annual medical charges: ${pred:,.2f}')

with right:
    st.subheader('Model overview')
    c1, c2, c3 = st.columns(3)
    c1.metric('Dataset rows', f"{metrics['sample_size']}")
    c2.metric('Train R²', f"{metrics['train_r2']:.3f}")
    c3.metric('Test R²', f"{metrics['test_r2']:.3f}")
    st.metric('Test RMSE (log scale)', f"{metrics['test_rmse_log']:.3f}")

    st.markdown('### Feature notes')
    st.write('- Charges are predicted from age, BMI, sex, children, smoker status, and region.')
    st.write('- Categorical values are one-hot encoded and numeric columns are standardized.')
    st.write('- Polynomial features help capture non-linear relationships in insurance charges.')

st.markdown('### Data preview')
st.dataframe(df.head(10), use_container_width=True)

