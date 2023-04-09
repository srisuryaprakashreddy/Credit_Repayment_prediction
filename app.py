import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load

nav = st.sidebar.radio("Navigation",["About","Predict"])

if nav=="About":
    st.title("Health Insurance Premium Predictor")
    st.text(" ")
    st.text(" ")
    st.image('images.png',width=600)

if nav=="Predict":
    st.title("Enter Details")
    creditpolicy='eligible'
    s=0
    
    creditpolicy = st.radio("Credit policy of the company  ",("eligible ","noteligible"))
    
    if (creditpolicy == "eligible"):
        s=0
    if (creditpolicy == "noteligible"):
        s=1
    st.text(s)

    intrestrate=st.slider("the intrest rate is ",0,130,11)
    st.text(intrestrate)

    # installments= st.number_input("installement plan to repay the loan in nomber of instalemets ",step=1,min_value=0)
    # st.text(installments)

    anuualincome=st.number_input("annual income",step=1,min_value=  1000)
    st.text(anuualincome)

    annualexpenses=st.number_input("enter his expenses",step=1,min_value=  100)
    st.text(annualexpenses)
    
    b=(annualexpenses/anuualincome)
        
    st.subheader("the debt-income-ratio : ")
    st.text(b)

    fibe=st.number_input("eneter the fibe credit score of the person ",step=1,min_value=450)
    
    # creditline=st.number_input("number of days with credit line",step=1,min_value=0)
    
    revolbal=st.number_input("The borrower’s revolving balance The borrower’s revolving balance",step=1,min_value=100)
    
    revolutil=st.number_input("The borrower’s revolving line utilization rate (the amount of the credit line used relative to total credit available).",step=1,min_value=100)
    
    inquirey=st.number_input("number of credit equenqiries in past 6 months",step=1,min_value=2)
    
    delinq=st.number_input("The number of times the borrower had been 30+ days past due on a payment in the past 2 years.",step=1,min_value=2)
    
    pubicimage=st.number_input("The borrower’s number of derogatory public records (bankruptcy filings, tax liens, or judgments).",step=1,min_value=1)

    purpose=st.radio("purpose of loan ",('credit_card',"debit_consolidation",'educational','home_improvement','major_purchase','small_bussiness'))
    
    st.text(purpose)
    credit_card=0
    debit_consolidation=0
    educational=0
    home_improvement=0
    major_purchase=0
    small_bussiness=0

    if (purpose==credit_card):
        credit_card=1
    if (purpose==debit_consolidation):
        debit_consolidation=1
    if (purpose==educational):
        educational=1
    if (purpose==home_improvement):
        home_improvement=1
    if (purpose==major_purchase):
        major_purchase=1
    if (purpose==small_bussiness):
        small_bussiness=1
    new_model = load_model('my_model_lending_club.h5')
    data=[[s,intrestrate,anuualincome,b,fibe,revolutil,inquirey,delinq,pubicimage,credit_card,debit_consolidation,educational,home_improvement,major_purchase,small_bussiness],
          [1,0.2164,14.52835448,29.96,827,119.0,33,13,5,1,1,1,1,1,1],[0,0.06,7.547501683,0.0,612,0.0,0,0,0,0,0,0,0,0,0]]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    later_scaler = load(open('scaler.pkl', 'rb'))
    later_model = load_model('my_model_lending_club.h5')
    

    if st.button("Predict"):
        st.subheader("Predicted that :")
        a =0.5
        a=new_model.predict(data[[0]])
        st.text(a)
    if a > 0.5:
        st.text("can calim loan")
    else:
        st.text("cannot claim loan")
