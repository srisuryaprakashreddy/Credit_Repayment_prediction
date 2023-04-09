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
    """    Credit scoring is a statistical analysis performed by lenders and financial 
    institutions to determine the ability of a person or a small, owner-operated 
    business to repay. Lenders use credit scoring to help decide whether to  extend 
    or deny credit as for any organization,even the slightest chance of financial risk
    can not be ignored or ruled out. The objective of this challenge is to create a 
    robust machine-learning model to predict which individuals are most likely to
    default on their loans,based on theirhistorical loan repayment behavior and 
    transactional activities. """)
    st.subheader("introduction")
    st.text(""" 
      Credit scoring is a crucial tool used by lenders and financial institutions to 
      assess the ability of individuals or small businesses to repay loans. By analyzing
      historical loan repayment behavior and transactional activities, lenders can make 
      informed decisions about extending or denying credit. In this project, our objective
      is to create a robust machine learning model that can accurately predict the likelihood
      of loan default. By doing so, we aim to help lenders minimize financial risk and make
      more informed decisions about extending credit.""")
    st.subheader("Dataset Used")
    df=pd.read_csv("loan_data.csv")
    st.dataframe(df)
    st.subheader("feature selection")
    st.text("""selected fetures are """)
    st.code("""
            credit.policy      if the customer meets the credit underwriting criteria of LendingClub.com, and “0” otherwise            
            int.rate           The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be riskier are assigned higher interest rates.            
            log.annual.inc     The natural log of the self-reported annual income of the borrower.
            dti                The debt-to-income ratio of the borrower (=amount of debt / annual income).           
            fico               The FICO credit score of the borrower.           
            revol.util         The borrower’s revolving line utilization rate (the amount of the credit line used relative to total credit available).           
            inq.last.6mths     The borrower’s number of inquiries by creditors in the last 6 months.           
            delinq.2yrs        The number of times the borrower had been 30+ days past due on a payment in the past 2 years.           
            pub.rec            The borrower’s number of derogatory public records (bankruptcy filings, tax liens, or judgments).           
            not.fully.paid                
            purpose_credit_card           
            purpose_debt_consolidation    
            purpose_educational           
            purpose_home_improvement      
            purpose_major_purchase        
            purpose_small_business       """)
    st.text("""from the above features "not.fully.paid selected" as target variable
     and remaining features are selected as independent features """)

    st.subheader("Preliminary Design")
    st.text("""

    1.Collect Data: Given the problem you want to solve,you will have to investigate 
    and obtain data that you will use to feed your machine.""")
    st.code(""" # import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential1
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from pickle import dump, load

%matplotlib inline
df = pd.read_csv('loan_data.csv')
""")
    st.text("""

    2.Prepare the data: Once you have collected your data,you will need to prepare 
    it for use in your model.""")
    st.code(""" 
    df_0 = df[df['not.fully.paid'] == 0]
    df_1 = df[df['not.fully.paid'] == 1]
    df_1_over = df_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_0, df_1_over], axis=0)
    final_data = pd.get_dummies(df_test_over,columns=cat_feats,drop_first=True)
    to_drop2 = ['revol.bal', 'days.with.cr.line', 'installment', 'revol.bal']
    final_data.drop(to_drop2, axis=1, inplace=True)
    to_train = final_data[final_data['not.fully.paid'].isin([0,1])]
to_pred = final_data[final_data['not.fully.paid'] == 2]
X = to_train.drop('not.fully.paid', axis=1).values
y = to_train['not.fully.paid'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)
    scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
    """)
    st.text("""
    
    3.Choose the model: There are many different types of models that you can use 
    for deep learning, and you will need to choose the one that is best suited 
    for your problem.""")
    st.code(""" model = Sequential()

model.add(
        Dense(94, activation='relu')
)

model.add(
        Dense(30, activation='relu')
)

model.add(
        Dense(15, activation='relu')
)


model.add(
        Dense(1, activation='sigmoid')
)

model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
)
""")
    st.text("""
    
    4.Train your machine model: Once you have chosen your model,you will need to 
    train it using your prepared data.""")
    ST.CODE("""
    early_stop = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=25
)

model.fit(
        X_train, 
        y_train, 
        epochs=200, 
        batch_size=256, 
        validation_data=(X_test, y_test),
         callbacks=[early_stop]
)""")
    st.text("""
    5.Evaluation: After training your model, you will need to evaluate its performance.""")
    st.code(""" predictions_new = (model_new.predict(X_test) >= 0.2).astype('int')

print(
        confusion_matrix(y_test,predictions_new), 
        '\n', 
        classification_report(y_test,predictions_new)
)""")
    st.text("""
    
    6.Prediction or Inference: Finally, once you have trained and evaluated your model,
      you can use it to make predictions or perform inference on new data. """)
    st.subheader("how to use the deployed model ")
    st.text("select prediction in the side slide bar ")
    st.code(""" 
            credit.policy                 
            int.rate                      
            log.annual.inc                
            dti                           
            fico                          
            revol.util                    
            inq.last.6mths                
            delinq.2yrs                   
            pub.rec 
            purpose_credit_card           
            purpose_debt_consolidation    
            purpose_educational           
            purpose_home_improvement      
            purpose_major_purchase        
            purpose_small_business        
    """)
    st.text("""give all required information above given and hit predict button""")

    st.subheader("Process of Deployment")
    st.text("""   
    I used Streamlit, a popular open-source framework, to deploy my machine learning
    models and data science projects.it was free of cost and purely in Python. 
    I deployed my Streamlit app on Streamlit Community Cloud which let me deploy
    my app in just one click.

    To deploy my app on Streamlit Community Cloud,I added my app code and dependencies
    to a GitHub repo. Streamlit Community Cloud launched my app directly from my GitHub
    repo. Then, I clicked “New app” from the upper right corner of my workspace, filled
    in my repo, branch,and file path, and clicked “Deploy”.""")
    st.subheader("streamlit code:-")
    st.code("""if nav=="Predict":
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
        st.text("cannot claim loan")""")
   

if nav=="Predict":
    st.title("Credit Repayment Predictor")
    creditpolicy='eligible'
    s=0
    st.subheader("Credit poilicy of Company")
    creditpolicy = st.radio("select   ",("Eligible for the policy of the company ","noteligible for the policy of  the company"))
    
    if (creditpolicy == "Eligible for the policy of the company"):
        s=0
    if (creditpolicy == "noteligible for the policy of  the company"):
        s=1
    st.text(s)
    st.subheader("intrest Rate")

    intrestrate=st.slider("Select Intrest",0,130,11)
    st.text(intrestrate)

    
    st.subheader("Anual Income")
    anuualincome=st.number_input("Annual Income",step=1,min_value=  1000)
    st.text(anuualincome)
    st.subheader("Anual Expenses")
    annualexpenses=st.number_input("Enter Expenses",step=1,min_value=  100)
    st.text(annualexpenses)
    
    b=(annualexpenses/anuualincome)
        
    st.subheader("The Debt-Income-Ratio : ")
    st.text(b)
    st.subheader("Credit score with reapective to the company")
    fibe=st.number_input(" ",step=1,min_value=450)
    
   
    st.subheader("Revolving Balance")
    revolbal=st.number_input(" ",step=1,min_value=100)
    st.subheader("Revolving Utilities")
    revolutil=st.number_input("the amount of the credit line used relative to total credit available",step=1,min_value=100)
    st.subheader("Enquiries")
    inquirey=st.number_input("number of credit equenqiries in past 6 months",step=1,min_value=2)
    st.subheader("deling")
    delinq=st.number_input("The number of times the borrower had been 30+ days past due on a payment in the past 2 years.",step=1,min_value=2)
    st.subheader("Public Records")
    pubicimage=st.number_input("The borrower’s number of derogatory public records (bankruptcy filings, tax liens, or judgments).",step=1,min_value=1)
    st.subheader("Purpose of loan")
    purpose=st.radio(" ",('credit_card',"debit_consolidation",'educational','home_improvement','major_purchase','small_bussiness'))
    
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
        st.text("eligible for  loan")
    else:
        st.text("not eligible for loan")
