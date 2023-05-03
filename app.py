import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
import mlem


def main():
    st.title("Giovanni's Summative Test")
    
    df=pd.read_csv('formart_house.csv')
    df = df.iloc[:506,:]
    df=df.astype(float)
    df=df.rename(columns={"medv":"price"})
    
    model = mlem.api.load('model_.mlem')
    X=df.drop('price',axis=1)
    y=df['price']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=667)
    y_pred = model.predict(X_test).round(2)
    
    df_pred=pd.DataFrame(data=list(zip(y_pred, y_test)),columns=['predicted', 'real'])
    df_pred['error'] = df_pred['predicted'] - df_pred['real']
    
    tab0,tab1,tab2,tab3 = st.tabs(['Modifica valori da predirre','Variazione di Prezzo','HeatMap Correlation Matrix','Error'])

    with tab0:
        modelup=mlem.api.load('model_.mlem')

        crim = st.number_input('Insert a crim', min_value=1,max_value=500)
        zn = st.number_input('Insert a zn',min_value=1,max_value=500)
        indus = st.number_input('Insert a indus',min_value=1,max_value=500)
        chas = st.number_input('Insert a chas',min_value=1,max_value=500) 
        nox = st.number_input('Insert a nox',min_value=1,max_value=500)
        rm = st.number_input('Insert a rm',min_value=1,max_value=500)
        age = st.number_input('Insert a age',min_value=1,max_value=500)
        dis = st.number_input('Insert a dis',min_value=1,max_value=500)
        rad = st.number_input('Insert a rad',min_value=1,max_value=500)
        tax = st.number_input('Insert a tax',min_value=1,max_value=500)
        ptratio = st.number_input('Insert a ptratio',min_value=1,max_value=500)
        b = st.number_input('Insert a b',min_value=1,max_value=500)
        lstat = st.number_input('Insert a lstat',min_value=1,max_value=500)   

        pred = modelup.predict([[crim, zn,indus, chas,nox,rm,age,dis,rad, tax,ptratio,b,lstat]]) [0]
        st.write(round(pred,1))
        
    with tab1:

        
        fig=plt.figure(figsize=(18,10))
        sns.histplot(data=df['price']) 
        #streamlit ha problemi nel renderizzare il pairplot
        # sns.pairplot(data=df[['price','crim','indus','age','nox']],hue='price')
        st.pyplot(fig)
    
    with tab2:
        fig2=plt.figure(figsize=(18,10))
        sns.heatmap(data=df.corr().round(1),annot=True)
        st.pyplot(fig2)
        
    with tab3:
        st.write('Confronto tra Target predetto e Target reale')
        st.dataframe(df_pred)
        
        lenght = y_pred.shape[0]
        x = np.linspace(0,lenght,lenght)

        figureError = plt.figure(figsize=(10,7))
        plt.plot(x,y_test,label='Y Reale')
        plt.plot(x,y_pred, label='Y Predetto') 
        plt.legend(loc=2)
        st.write('Plot degli errori')
        st.pyplot(figureError)
        
if __name__ == "__main__":
    main()