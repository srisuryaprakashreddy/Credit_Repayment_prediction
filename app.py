import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import pandas as pd



nav = st.sidebar.radio("Navigation",["About the project","Predict"])

if nav=="About the project":
    st.title("Predicting Loan Defaults using Deep Learning with Keras & Tensorflow")
    st.subheader("Problem Statement:")
    st.text(
    """    Credit scoring is a statistical analysis performed by 
    lenders and financial institutions to determine the 
    ability of a person or a small, owner-operated business
    to repay. Lenders use credit scoring to help decide 
    whether to  extend or deny credit as for any organization,
    even the slightest chance of financial risk can not be 
    ignored or ruled out. The objective of this challenge 
    is to create a robust machine-learning model to predict 
    which individuals are most likely to default on their loans,
    based on theirhistorical loan repayment behavior and 
    transactional activities. """)
    st.subheader("introduction")
    st.text(""" 
      Credit scoring is a crucial tool used by lenders and 
      financial institutions to assess the ability of
      individuals or small businesses to repay loans. 
      By analyzing historical loan repayment behavior and 
      transactional activities, lenders can make informed 
      decisions about extending or denying credit. In this 
      project, our objective is to create a robust machine 
      learning model that can accurately predict the likelihood
      of loan default. By doing so, we aim to help lenders
      minimize financial risk and make more informed decisions 
      about extending credit.""")
    st.subheader("Dataset Used")
    df=pd.read_csv("loan_data.csv")
    st.dataframe(df)
    st.subheader("feature selection")
    st.text("""selected fetures are """)
    st.code("""
credit.policy                 0.0
int.rate                      0.0
log.annual.inc                0.0
dti                           0.0
fico                          0.0
revol.util                    0.0
inq.last.6mths                0.0
delinq.2yrs                   0.0
pub.rec                       0.0
not.fully.paid                0.0
purpose_credit_card           0.0
purpose_debt_consolidation    0.0
purpose_educational           0.0
purpose_home_improvement      0.0
purpose_major_purchase        0.0
purpose_small_business        0.0""")
    st.text("""from the above features not.fully.paid selected as target variable
     and remaining features are selected as independent features """)

    st.subheader("Preliminary Design")
    st.text("""

    1.Collect Data: Given the problem you want to solve, 
    you will have to investigate and obtain data that you 
    will use to feed your machine.

    2.Prepare the data: Once you have collected your data,
    you will need to prepare it for use in your model.
    
    3.Choose the model: There are many different types of 
    models that you can use for deep learning, and you will
    need to choose the one that is best suited for your problem.
    
    4.Train your machine model: Once you have chosen your model,
    you will need to train it using your prepared data.
    
    5.Evaluation: After training your model, you will need to 
    evaluate its performance.
    
    6.Parameter Tuning: You may need to adjust the parameters of
    your model to improve its performance.
    
    7.Prediction or Inference: Finally, once you have trained and
    evaluated your model, you can use it to make predictions or 
    perform inference on new data. """)
   
   

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

    

    anuualincome=st.number_input("annual income",step=1,min_value=  1000)
    st.text(anuualincome)

    annualexpenses=st.number_input("enter his expenses",step=1,min_value=  100)
    st.text(annualexpenses)
    
    b=(annualexpenses/anuualincome)
        
    st.subheader("the debt-income-ratio : ")
    st.text(b)

    fibe=st.number_input("eneter the fibe credit score of the person ",step=1,min_value=450)
    
   
    
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
    a =0.5
    
    if st.button("Predict"):
        st.subheader("Predicted that :")
        
        a=new_model.predict(data[[0]])
        st.text(a)
    if a > 0.5:
        st.text("can calim loan")
    else:
        st.text("cannot claim loan")
