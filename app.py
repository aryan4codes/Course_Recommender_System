import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import streamlit as st

data= pd.read_csv('\Coursera.csv')

data = data[['Course_Name','Difficulty Level','Course Description','Skills']]
# Removing spaces between the words (Lambda funtions can be used as well)

data['Course_Name'] = data['Course_Name'].str.replace(' ',',')
data['Course_Name'] = data['Course_Name'].str.replace(',,',',')
data['Course_Name'] = data['Course_Name'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace(' ',',')
data['Course Description'] = data['Course Description'].str.replace(',,',',')
data['Course Description'] = data['Course Description'].str.replace('_','')
data['Course Description'] = data['Course Description'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace('(','')
data['Course Description'] = data['Course Description'].str.replace(')','')

#removing paranthesis from skills columns 
data['Skills'] = data['Skills'].str.replace('(','')
data['Skills'] = data['Skills'].str.replace(')','')
data['tags'] = data['Course_Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']

new_df = data[['Course_Name','tags']]

new_df['tags'] = data['tags'].str.replace(',',' ')
new_df['Course_Name'] = data['Course_Name'].str.replace(',',' ')
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) #lower casing the tags column


#Text Vectorization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#defining the stemming function
def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)


new_df['tags'] = new_df['tags'].apply(stem) #applying stemming on the tags column


#Similarity Measure
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

#Recommend Function

def recommend(user_text):
    user_text = stem(user_text.lower())
    user_vector = cv.transform([user_text]).toarray()

    if user_vector.sum() == 0:
        st.warning("No recommendations available for the entered text.")
        return []

    distances = cosine_similarity(user_vector, vectors)[0]

    if not any(distances):
        st.warning("No recommendations available for the entered text.")
        return []

    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[:10]
    recommended_courses = [new_df.iloc[i[0]]['Course_Name'] for i in course_list]
    return recommended_courses if recommended_courses else []


st.title("Course Recommender System by DODS - Powered by Coursera")
user_input = st.text_input("Enter your text:", "")

if st.button("Get Recommendations"):
    recommendations = recommend(user_input)
    st.success("Top 10 Recommended Courses:")
    for i, rec_course in enumerate(recommendations, start=1):
        st.write(f"{i}. {rec_course}")

