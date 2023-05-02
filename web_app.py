# importing relevant python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# preprocessing
import requests
from flask import Flask, jsonify
from flask import request
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
# modeling
from sklearn import svm
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px


load_dotenv()

print(os.getenv("HUGGINGFACE_API"))
print(st.secrets["HUGGINGFACE_API"])
API_URL_HATE = "https://api-inference.huggingface.co/models/IMSyPP/hate_speech_en"
headers = {"Authorization": "Bearer " +  st.secrets["HUGGINGFACE_API"] }

def query_hate(payload):
	response = requests.post(API_URL_HATE, headers=headers, json=payload)
	return response.json()

# creating page sections
site_header = st.container()
business_context = st.container()
data_desc = st.container()
performance = st.container()
tweet_input = st.container()
model_results = st.container()
sentiment_analysis = st.container()
hate_speech=st.container()
contact = st.container()

with site_header:
    st.title('Hate Speech Detection & Analysis')
    st.write("""
    
    This project aims to **automate content moderation** to identify hate speech using **machine learning binary classification algorithms.** 
    
    Baseline models included Random Forest, Naive Bayes, Logistic Regression and Support Vector Machine (SVM). The final model was a **Logistic Regression** model that used Count Vectorization for feature engineering. It produced an F1 of 0.3958 and Recall (TPR) of 0.624.  

    """)

with business_context:
    st.header('The Problem of Content Moderation')
    st.write("""
    
    **Human content moderation exploits people by consistently traumatizing and underpaying them.** In 2019, an [article](https://www.theverge.com/2019/6/19/18681845/facebook-moderator-interviews-video-trauma-ptsd-cognizant-tampa) on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, **every major tech company**, including **Twitter**, uses human moderators to some extent, both domestically and overseas.
    
    Hate speech is defined as **abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation.**  Usually, the difference between hate speech and offensive language comes down to subtle context or diction.
    
    """)

with data_desc:
    understanding, venn = st.columns(2)
    with understanding:
        st.text('')
        st.write("""
        The **data** for this project was sourced from a Cornell University [study](https://github.com/t-davidson/hate-speech-and-offensive-language) titled *Automated Hate Speech Detection and the Problem of Offensive Language*.
        
        The `.csv` file has **24,802 rows** where **6% of the tweets were labeled as "Hate Speech".**

        Each tweet's label was voted on by crowdsource and determined by majority rules.
        """)
    with venn:
        st.image(Image.open('visualizations/word_venn.png'), width = 400)

with performance:
    description, conf_matrix = st.columns(2)
    with description:
        st.header('Final Model Performance')
        st.write("""
        These scores are indicative of the two major roadblocks of the project:
        - The massive class imbalance of the dataset
        - The model's inability to identify what constitutes as hate speech
        """)
    with conf_matrix:
        st.image(Image.open('visualizations/normalized_log_reg_countvec_matrix.png'), width = 400)

with tweet_input:
    st.header('Is Your Tweet Considered Hate Speech?')
    st.write("""*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
    # user input here
    user_text = st.text_input('Enter text', max_chars=280) # setting input as user_text

with model_results:    
    st.subheader('Prediction:')
    if user_text:
    # processing user_text
        # removing punctuation
        user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
        # tokenizing
        stop_words = list(stopwords.words('english'))
        tokens = nltk.word_tokenize(user_text)
        # removing stop words
        stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
        # taking root word
        lemmatizer = WordNetLemmatizer() 
        lemmatized_output = []
        for word in stopwords_removed:
            lemmatized_output.append(lemmatizer.lemmatize(word))

        # instantiating count vectorizor
        count = CountVectorizer(stop_words=stop_words)
        X_train = pickle.load(open('pickle/X_train_2.pkl', 'rb'))
        X_test = lemmatized_output
        X_train_count = count.fit_transform(X_train)
        X_test_count = count.transform(X_test)

        # loading in model
        final_model = pickle.load(open('pickle/final_log_reg_count_model.pkl', 'rb'))

        # apply model to make predictions
        prediction = final_model.predict(X_test_count[0])
        analyzer = SentimentIntensityAnalyzer() 
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text) 
        if sentiment_dict['compound'] >= 0.05 : 
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05 : 
            category = ("**Negative 🚫**") 
        else : 
            category = ("**Neutral ☑️**")
        print(category)
        def pred(category):
            if category == "**Negative 🚫**":
                st.subheader('**Hate Speech**')
            else:
                st.subheader('**Not Hate Speech**')
            st.text('')
        pred(category)
        

with sentiment_analysis:
    if user_text:
        st.header('Sentiment Analysis with VADER')
        
        # explaining VADER
        st.write("""*VADER is a lexicon designed for scoring social media. More information can be found [here](https://github.com/cjhutto/vaderSentiment).*""")
        # spacer
        st.text('')
    
        # instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer() 
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text) 
        if sentiment_dict['compound'] >= 0.05 : 
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05 : 
            category = ("**Negative 🚫**") 
        else : 
            category = ("**Neutral ☑️**")
        print(category)
        pred(category)

        # score breakdown section with columns
        breakdown, graph = st.columns(2)
        with breakdown:
            # printing category
            st.write("Your Tweet is rated as", category) 
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**") 
            st.write(sentiment_dict['neg']*100, "% Negative") 
            st.write(sentiment_dict['neu']*100, "% Neutral") 
            st.write(sentiment_dict['pos']*100, "% Positive") 
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph) 

with hate_speech:
    st.header('Classification of comments')
    sentence = st.text_input('Enter social media comment','war in ukraine', max_chars=280) 
    def hate_en():
        print(sentence)
        output=query_hate({
        "inputs": str(sentence)})
        # print(output[0])
        result = {}
        if sentence:
            for data in output[0]:
                if data['label'] == "LABEL_0":
                    result["ACCEPTABLE"] = round(data['score']*100, 2)
                elif data['label'] == "LABEL_1":
                    result["INAPPROPRIATE"] = round(data['score']*100, 2)
                elif data['label'] == "LABEL_2":
                    result["OFFENSIVE"] = round(data['score']*100, 2)
                elif data['label'] == "LABEL_3":
                    result["VIOLENT"] = round(data['score']*100, 2)

            labels = list(result.keys())
            values = list(result.values())
            print(result)

        # Map labels to their respective categories
            labels = ["Acceptable" if label == "ACCEPTABLE" else "Inappropriate" if label == "INAPPROPRIATE" else "Offensive" if label == "OFFENSIVE" else "Violent" for label in labels]


            # # Use `hole` to create a donut-like pie chart
            # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
            # # fig.show()
            # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            # print(graphJSON)
            # print(type(fig))
            # return graphJSON
            return result
        else:
            pass
   
    hatespeech=hate_en() 
    print(hatespeech)

	# create a list of labels and values
    labels = list(hatespeech.keys())
    values = list(hatespeech.values())

    fig = px.pie(values=values, names=labels)

	# display the pie chart using st.plotly_chart
    st.plotly_chart(fig)

with contact:
    st.markdown("---")
    st.header('For More Information')
    st.text('')
    st.write("""


    Contact Prathik Shetty via [prathikkshetty15@gmail.com](mailto:prathikkshetty15@gmail.com).
    """)

    st.subheader("Let's Connect!")
    st.write("""
    
     [Github](https://github.com/prathikshetty2002)  |  [Twitter](https://twitter.com/I_am_prathik)


    """)
