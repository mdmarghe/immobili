import streamlit as st
import pandas as pd
import numpy as np
from numpy import dtype
import joblib


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.istockphoto.com/fr/photo/image-répétable-sans-couture-avec-revêtement-en-plastique-vert-texture-de-clôture-gm504882450-83391459");
             background-attachment: fixed;
             background-size: cover
        }}
         </style>
         """,
        unsafe_allow_html=True
    )
add_bg_from_url()

def main():
    newmodel = joblib.load('regression_test.pkl')
    st.title('Analisi dati immobili')
    st.subheader('basta un click')
    csv=st.file_uploader("porta qui i tuoi dati e avrai la previsione del prezzo")
    if csv is not None:
        df=pd.read_csv(csv)
        st.dataframe(df)
        prediction=newmodel.prediction(csv)
        st.dataframe(prediction)
        
        
        
    st.subheader('Oppure inserisci qui i tuoi dati')
    x1 = st.slider('crim', min_value=0, max_value=500, value=300)
    x2= st.slider('zn', min_value=0, max_value=500, value=300)
    x3 = st.slider('indus', min_value=0, max_value=22, value=7)
    x4 = st.slider('chas', min_value=0, max_value=500, value=300)
    x5 = st.slider('nox', min_value=0, max_value=500, value=300)
    x6 = st.slider('rm', min_value=0, max_value=500, value=300)
    x7 = st.slider('age', min_value=0, max_value=500, value=300)
    x8 = st.slider('dis', min_value=0, max_value=500, value=300)
    x9 = st.slider('rad', min_value=0, max_value=500, value=300)
    x10 = st.slider('tax', min_value=0, max_value=500, value=300)
    x11=st.slider('ptratio', min_value=1, max_value=711, value=300)
    x12=st.slider('b', min_value=0, max_value=500, value=300)
    x13=st.slider('lstat', min_value=0, max_value=60, value=50)
    
    input_data = {'crim': x1,
                  'zn': x2,
                  'indus': x3,
                  'chas': x4,
                  'nox': x5,
                  'rm': x6,
                  'age': x7,
                  'dis': x8,
                  'rad': x9,
                  'tax': x10,
                  'ptratio': x11,
                  'b': x12,
                  'lstat': x13,
                  }

    input_df = pd.DataFrame(input_data, index=[0])
    prediction = newmodel.predict([[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]])
    st.write('la previsione del tuo prezzo è', prediction)

if __name__ == "__main__":
    main()
    
#streamlit run app.py