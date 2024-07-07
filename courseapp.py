# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:54:28 2024

@author: HP
"""

import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import os

# Assuming course_model.sav is in the same directory as courseapp.py
model_path = os.path.join(os.path.dirname(__file__), 'course_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))


#loaded_model= pickle.load(open('C:/Users/HP/Desktop/online course engagement dataset/course_model.sav', 'rb'))

def course_prediction(input_data):
    
    
    CourseCategory_encoder= LabelEncoder()
    
    
    CourseCategory_encoder.fit(['Health', 'Arts', 'Science', 'Programming', 'Business'])
    
  

    processed_data = []
    for feature, value in input_data.items():
        if feature == "CourseCategory":
            processed_data.append(CourseCategory_encoder.transform([value])[0])
        else:
            try:
                processed_data.append(float(value))
            except ValueError:
                processed_data.append(0)
                
    processed_data = np.array(processed_data).reshape(1, -1)
    
    return processed_data
    

def main():
    st.title('Course completion prediction')
    
    UserID= st.text_input('Unique identifier for each user')
    CourseCategory=st.selectbox('CourseCategory',['Health', 'Arts', 'Science', 'Programming', 'Business'])
    TimeSpentOnCourse=st.text_input('Total time spent by the user on the course in hours')
    NumberOfVideosWatched= st.text_input('Total number of videos watched by the user')
    NumberOfQuizzesTaken= st.text_input('Total number of quizzes taken by the user')
    QuizScores= st.text_input('Average scores achieved by the user in quizzes (percentage)')
    CompletionRate= st.text_input('Percentage of course content completed by the user')
    DeviceType= st.selectbox('Type of device used by the user (e.g., Desktop, Mobile)', ['Desktop', 'Mobile'])
    
    
    diagnosis=''
    
    if st.button('CourseCompletion'):
        input_data = {
            "UserID": UserID,
            "CourseCategory": CourseCategory,
            "TimeSpentOnCourse": TimeSpentOnCourse,
            "NumberOfVideosWatched": NumberOfVideosWatched,
            "NumberOfQuizzesTaken": NumberOfQuizzesTaken,
            "QuizScores": QuizScores,
            "CompletionRate": CompletionRate,
            "DeviceType": DeviceType
        }
        processed_data = course_prediction(input_data)
    
        prediction= loaded_model.predict(processed_data)
        if (prediction[0]==0):
            diagnosis='Not Completed'
        else:
            diagnosis='Completed'
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
