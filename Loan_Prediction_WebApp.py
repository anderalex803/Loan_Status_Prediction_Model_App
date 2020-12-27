#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle



Extra_Tree_Model=pickle.load(open('Extra_Tree_Model.pkl','rb'))
log_model=pickle.load(open('log_model.pkl','rb'))
DT_model=pickle.load(open('DT_model.pkl','rb'))



def classify(Credit_History):
    if Credit_History == 0:
        return 'No, sorry your loan was not approved'
    else:
        return 'Yes, your loan was approved'
def main():
    st.title("Loan Status Prediction")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Status Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Extra Tree','Logistic Regression','Decision Tree']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    gen=st.button('Select Gender', Male, Female)
    mar=st.button('Select Married', Yes, No)
    dep=st.button('Select Dependents', Yes, No)
    edu=st.slider('Select Education', 0, 3)
    self=st.button('Select Self_Employed', Yes, No)
    ai=st.slider('Select ApplicantIncome', 0, 81000)
    cai=st.slider('Select CoapplicantIncome', 0, 41667)
    ch=st.button('Select Credit_History', 0, 1)
    la=st.slider('Select LoanAmount', 0, 700)
    lat=st.slider('Select Loan_Amount_Term', 0, 480)
    pa=st.button('Select Property_Area', Rural, Semiurban, Urban)
    inputs=[[gen,mar,dep,edu,self,ai,cai,ch,la,lat,pa]]
    if st.button('Classify'):
        if option=='Extra Tree Model':
            st.success(classify(Extra Tree Model.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))
        else:
           st.success(classify(DT_model.predict(inputs)))


if __name__=='__main__':
    main()

