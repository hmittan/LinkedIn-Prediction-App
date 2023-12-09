import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

with st.form(key='my_form_to_submit'):
    st.markdown("# LinkedIn User Determination App")
    st.markdown("### Answer the following questions and await the prediction!")
    person_pred = np.zeros(6)

    # Income
    income = st.number_input("#1. What is your annual household income? (No $ sign)")

    if income > 150000:
        person_pred[0] = 9
    elif income > 100000:
        person_pred[0] = 8
    elif income > 75000:
        person_pred[0] = 7   
    elif income > 50000:
        person_pred[0] = 6  
    elif income > 40000:
        person_pred[0] = 5   
    elif income > 30000:
        person_pred[0] = 4   
    elif income > 20000:
        person_pred[0] = 3   
    elif income > 10000:
        person_pred[0] = 2   
    else: person_pred[0] = 1

    # Age
    person_pred[1] = st.number_input("#2. How old are you?")

    # Married
    married_input = st.selectbox("#3. Are you married?",
    ("Yes", "No"), index=None, placeholder="Select...")

    if married_input is "Yes":
        person_pred[2] = 1
    elif married_input is "No":
        person_pred[2] = 0

    # Parent
    parent_input = st.selectbox("#4. Are you a parent?",
    ("Yes", "No"), index=None, placeholder="Select...")

    if parent_input is "Yes":
        person_pred[3] = 1
    elif parent_input is "No":
        person_pred[3] = 0

    # Gender
    gender_input = st.selectbox(
        "#6. Are you a male or female?",
        ("Male", "Female"), index=None, placeholder="Select...")

    if gender_input is "Female":
        person_pred[4] = 1
    elif gender_input is "Male":
        person_pred[4] = 0

    # Education
    st.write("Use the following information for Question 6:")
    st.write("1 - Less than high school")
    st.write("2 - High school incomplete")
    st.write("3 - High school graduate")
    st.write("4 - Some college, no degree")
    st.write("5 - Two-year associate degree")
    st.write("6 - Four-year Bachelor's degree")
    st.write("7 - Postgraduate or professional degree")
    st.write("8 - Postgraduate or professional degree")

    person_pred[5] = st.selectbox(
    "#6. What is the highest level of school/degree you've completed?",
    (1, 2, 3, 4, 5, 6, 7, 8), index=None, placeholder="Select corresponding number from above...")
    submit_button = st.form_submit_button(label='Submit')



if submit_button:
    s = pd.read_csv("social_media_usage.csv")

    def clean_sm(x):
        x1 = np.where(x == 1, 1, 0)
        return x1
        
    ss = pd.DataFrame({
        "sm_li":clean_sm(s["web1h"]),
        "income":np.where(s["income"] > 9, np.nan, s["income"]),
        "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
        "parent":np.where(s["par"] == 1, 1, 0),
        "married":np.where(s["marital"] == 1, 1, 0),
        "female":np.where(s["gender"] == 2, 1, 0),
        "age":np.where(s["age"] > 98, np.nan, s["age"])})

    ss = ss[np.isfinite(ss).all(1)]

    y = ss["sm_li"]
    X = ss[["income", "age", "married", "parent", "female", "education"]]

    X_train,X_test,y_train,y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,   
                                                    random_state=123)
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(X_train, y_train)

    person1 = list(person_pred)
    predicted_class1 = lr.predict([person1])
    probs1 = lr.predict_proba([person1])

    if predicted_class1[0] == 0:
        st.markdown(f"#### This user is predicted to not be a LinkedIn user (Predicted class: {predicted_class1[0]})")
    else: st.markdown(f"#### This user is predicted to be a LinkedIn user (Predicted class: {predicted_class1[0]})")

    st.markdown(f"#### The probability that this user is a LinkedIn user is {round(probs1[0][1],4)}")


