import pandas as pd
import numpy as np
import itertools
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle
import time

#Load data
dir = "Data/hungarian.data"
with open(dir, encoding='Latin1') as file:
  lines = [line.strip() for line in file]

data = itertools.takewhile(
    lambda x: len(x) == 76,
    (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)

df = df.iloc[:,:-1]
df = df.drop(df.columns[0], axis=1)
df = df.astype(float)

#Validasi data
df.replace(-9.0, np.nan, inplace=True)

#Menenteukan object data
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

column_mapping = {
    2: 'age',
    3: 'sex',
    8: 'cp',
    9: 'trestbps',
    11: 'chol',
    15: 'fbs',
    18: 'restecg',
    31: 'thalach',
    37: 'exang',
    39: 'oldpeak',
    40: 'slope',
    43: 'ca',
    50: 'thal',
    57: 'target'
}

df_selected.rename(columns=column_mapping, inplace=True)

#Membersihkan data
columns_to_drop = ['ca', 'slope', 'thal']

df_selected = df_selected.drop(columns_to_drop, axis=1)

meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanRestCG = meanRestCG.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)

meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanRestCG = round(meanRestCG.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())

fill_values = {'trestbps': meanTBPS, 'chol': meanChol, 'fbs': meanfbs,
                'thalach':meanthalach,'exang':meanexang,'restecg':meanRestCG}


df_clean = df_selected.fillna(value=fill_values)
df_clean.drop_duplicates(inplace=True)

X = df_clean.drop("target", axis=1)
y = df_clean.iloc[:,-1]

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

model = pickle.load(open("Model/xgb_model.pkl", 'rb'))

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y


# STREAMLIT
st.set_page_config(
  page_title = "Hungarian Heart Disease",
  page_icon = ":heart:"
)

st.title("Hungarian Heart Disease")
st.write(f"**_Model's Accuracy_** :  :green[**{accuracy}**]% (:red[_Do not copy outright_])")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
  sex_dict = {"Male": 0, "Female": 1}
  cp_dict = {"typical angina": 1, "atypical angina": 2, "non-anginal pain": 3, "asymptomatic": 4}
  fbs_dict = {"True": 1, "False": 0}
  restecg_dict = {"normal": 0, "having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)": 1, "showing probable or definite left ventricular hypertrophy by Estes criteria": 2}
  exang_dict = {"Yes": 1, "No": 0}

  age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
  sex = sex_dict[st.selectbox("Sex", options=["Male", "Female"])]
  cp = cp_dict[st.selectbox("Chest pain type", options=["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])]
  trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=0, max_value=300, value=120, step=1)
  chol = st.number_input("Serum cholesterol (mg/dl)", min_value=0, max_value=1000, value=200, step=1)
  fbs = fbs_dict[st.selectbox("Fasting blood sugar > 120 mg/dl", options=["True", "False"])]
  restecg = restecg_dict[st.selectbox("Resting electrocardiographic results", options=["normal", "having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)", "showing probable or definite left ventricular hypertrophy by Estes criteria"])]
  thalach = st.number_input("Maximum heart rate achieved", min_value=0, max_value=300, value=150, step=1)
  exang = exang_dict[st.selectbox("Exercise induced angina", options=["Yes", "No"])]
  oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

  predict_btn = st.button("Predict", type="primary")

  result = ":violet[-]"

  # prediction = "[-]"

  # def display_result(prediction):
  #   if prediction == 0:
  #     st.success("The diagnosis of heart disease is negative.")
  #   else:
  #     st.error("The diagnosis of heart disease is positive.")

  if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = model.predict(inputs)[0]
    #display_result(prediction)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
        status_text.text(f"{i}% complete")
        bar.progress(i)
        time.sleep(0.01)
        if i == 100:
          time.sleep(1)
          status_text.empty()
          bar.empty()

    if prediction == 0:
        result = ":green[**Healthy**]"
    elif prediction == 1:
        result = ":orange[**Heart disease level 1**]"
    elif prediction == 2:
        result = ":orange[**Heart disease level 2**]"
    elif prediction == 3:
        result = ":red[**Heart disease level 3**]"
    elif prediction == 4:
        result = ":red[**Heart disease level 4**]"

    st.write("")
    st.write("")
    st.subheader("Prediction:")
    st.subheader(result)

with tab2:
  st.header("Predict multiple data:")

  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Heart disease level 1"
      elif prediction == 2:
        result = "Heart disease level 2"
      elif prediction == 3:
        result = "Heart disease level 3"
      elif prediction == 4:
        result = "Heart disease level 4"
      result_arr.append(result)

    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)