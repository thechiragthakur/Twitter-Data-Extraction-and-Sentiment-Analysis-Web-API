#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm
import pycountry

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#sns.set_style('darkgrid')


STYLE = """
<style="background-color:orange;">
img {
    max-width: 100%;
}
</style> """

def main():
    """ Common ML Dataset Explorer """
    #st.title("Live twitter Sentiment analysis")
    #st.subheader("Select a topic which you'd like to get the sentiment analysis on :")
    


    html_temp = """
	<div style="background-color:orange;"><p style="color:black;font-size:40px;padding:9px">Twitter Analyzer </p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Select a topic for Sentiment Analysis :")

    ################# Twitter API Connection #######################
    consumer_key= 'QLdhEgbqKEZCSVsW0kedh4cXi'
    consumer_secret= '1SwSssnHayK8c1bdsiDksMhvaztv9rDQ4cva2NM7VTafXQb9Yr'

    access_token= '1441295872852066306-mINc6c8y3TwGaUEV8ytUdqkipK6lua'
    access_token_secret= 'W6oC0kDfLjucXVPDJm8ZTtrQPxgYqmw5G922SS0SEGGBg'



    # Use the above credentials to authenticate the API.

    auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
    auth.set_access_token( access_token , access_token_secret )
    api = tweepy.API(auth)
    ################################################################
    
    df = pd.DataFrame(columns=["Date","User","Tweet","Likes","RT",'User_location','id'])
    
                          
    # Write a Function to extract tweets:
    def get_tweets(Topic,Count):
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(api.search, q=Topic,count=100, lang="en",exclude='retweets').items():
            #time.sleep(0.1)
            #my_bar.progress(i)
            df.loc[i,"Date"] = tweet.created_at
            df.loc[i,"User"] = tweet.user.name
            df.loc[i,"id"] = tweet.id
            df.loc[i,"IsVerified"] = tweet.user.verified
            df.loc[i,"Tweet"] = tweet.text
            df.loc[i,"Likes"] = tweet.favorite_count
            df.loc[i,"RT"] = tweet.retweet_count
            df.loc[i,"User_location"] = tweet.user.location
            #df.to_csv("TweetDataset.csv",index=False)
            #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i=i+1
            if i>Count:
                break
            else:
                pass
            
            
    # Function to Clean the Tweet.
    def clean_tweet(tweet):
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())   


    def process_tweets(tweet):
    
    #remove emoji
        tweet =re.sub( "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+",'', str(tweet),flags=re.MULTILINE) # no emoji


        # Remove links
        tweet = re.sub(r"http\S+", "", tweet,flags=re.MULTILINE)

        # Remove mentions and hashtag
        tweet = re.sub(r'\@\w+|\#','', tweet)


        # Tokenize the words
        tokenized = word_tokenize(tweet)

        # Remove the stop words
        tokenized = [token for token in tokenized if token not in stopwords.words("english")] 

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        tokenized = [lemmatizer.lemmatize(token, pos='a') for token in tokenized]

        # Remove non-alphabetic characters and keep the words contains three or more letters
        tokenized = [token for token in tokenized if token.isalpha() and len(token)>2]

        return tokenized
    
    # Funciton to analyze Sentiment
    def analyze_sentiment(polarity):

        if polarity > 0:
            return "Positive"
        if polarity == 0:
            return "Neutral"
        if polarity < 0:
            return "Negative"
    
    def get_countries(location):
    
        # If location is a country name return its alpha2 code
        if pycountry.countries.get(name= location):
            return pycountry.countries.get(name = location).alpha_2

        # If location is a subdivisions name return the countries alpha2 code
        try:
            pycountry.subdivisions.lookup(location)
            return pycountry.subdivisions.lookup(location).country_code
        except:
            # If the location is neither country nor subdivision return the "unknown" tag
            return "unknown"


        
   
        

    # Collect Input from user :
    Topic = str()
    Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))     
    
    if len(Topic) > 0 :
        
        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Tweets are being extracted"):
            get_tweets(Topic , Count=1000)
        st.success('Tweets have been Extracted !!!!') 
        

        df["clean_tweet"] = df["Tweet"].apply(lambda tweet : clean_tweet(tweet))

        # Call the function and store the result into a new column
        df["Processed"] = df["Tweet"].str.lower().apply(lambda tweet : process_tweets(tweet))
        
        #df["clean_tweet"] = df["Tweet"].apply(lambda x : clean_tweet(x)
        
        
        df["Polarity"] = df["Processed"].apply(lambda word: TextBlob(str(word)).sentiment.polarity)
        
        
        df["Subjectivity"] = df["Processed"].apply(lambda word: TextBlob(str(word)).sentiment.subjectivity)
        
         # Call function to get the Sentiments
        df["Sentiment"] = df["Polarity"].apply(lambda x : analyze_sentiment(x))
        
        df["Country"] = df["User_location"].apply(get_countries)
        
        flat_list = [item for sublist in df["Processed"] for item in sublist]
        # Create our contextual stop words
        
        tfidf_stops =["cash"
        ,"compensation"
        ,"earnings"
        ,"interest"
        ,"livelihood"
        ,"pay"
        ,"proceeds"
        ,"profit"
        ,"revenue"
        ,"royalty"
        ,"salary"
        ,"wage"
        ,"assets"
        ,"avails"
        ,"benefits"
        ,"commission"
        ,"dividends"
        ,"drawings"
        ,"gains"
        ,"gravy"
        ,"gross"
        ,"harvest"
        ,"honorarium"
        ,"means"
        ,"payoff"
        ,"receipts"
        ,"returns"]




        # Initialize a Tf-idf Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, stop_words= tfidf_stops)
        # Fit and transform the vectorizer
        tfidf_matrix = vectorizer.fit_transform(flat_list)
        # Let's see what we have
        #tfidf_matrix
        # Create a DataFrame for tf-idf vectors and display the first rows
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns= vectorizer.get_feature_names())
        tweets_processed=tfidf_df.head()
        #tweets_processed
        
        
       
        
        # Write Summary of the Tweets
        #st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
        #st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"])))
        #st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"]=="Negative"])))
        #st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))


    #Activity_choice = st.selectbox("Select the Activities",  ["See the Extracted Data","Get Count Plot for Different Sentiments" ,"Get Pie Chart for Different Sentiments"])
        
    

    st.sidebar.title("About Ztudium")
    from PIL import Image
    image = Image.open('logo1.png')
    st.sidebar.image(image,use_column_width=True)
    st.sidebar.info("Ztudium is a global leading maker of industry 4IR Fourth Industrial Revolution technologies and research. We build software, research products and service solutions, using blockchain, AI and digital transformation DNA.")

    st.sidebar.subheader("Tweet Analyzer") 

    tw1=st.sidebar.radio("Summary of the Tweets", ["None","Total Tweets","Positive Tweets","Negative Tweets","Neutral Tweets"])

    if  tw1=="Total":
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))

    if tw1=="Positive":
        st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"])))


    if tw1=="Negative":
        st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"]=="Negative"])))


    if tw1=="Neutral":
        st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))

        #st.write("Total Negative Tweets are : {}".(len(df[df["Sentiment"]=="Negative"])))





    
    Activity_choice = st.sidebar.radio("Select the Activities",  ["None","See the Extracted Data","Get Count Plot for Different Sentiments","Get Pie Chart for Different Sentiments"])
        
        # See the Extracted Data : 
    if Activity_choice=="See the Extracted Data":
        #st.markdown(html_temp, unsafe_allow_html=True)
        st.success("Below are the Extracted Data :")
        st.write(df.head(5))

    if Activity_choice=="Get Count Plot for Different Sentiments":

        st.success("Generating A Count Plot")
        st.subheader(" Count Plot for Different Sentiments")
        st.write(sns.countplot(df["Sentiment"]))
        st.pyplot()

        # Piechart 
    if Activity_choice=="Get Pie Chart for Different Sentiments":
        st.success("Generating A Pie Chart")
        a=len(df[df["Sentiment"]=="Positive"])
        b=len(df[df["Sentiment"]=="Negative"])
        c=len(df[df["Sentiment"]=="Neutral"])
        d=np.array([a,b,c])
        explode = (0.1, 0.0, 0.1)
        st.write(plt.pie(d,shadow=True,explode=explode,labels=["Positive","Negative","Neutral"],autopct='%1.2f%%'))
        st.pyplot()

 


    #st.sidebar.subheader("Scatter-plot setup")
    #box1 = st.sidebar.selectbox(label= "X axis", options = numeric_columns)
    #box2 = st.sidebar.selectbox(label="Y axis", options=numeric_columns)
    #sns.jointplot(x=box1, y= box2, data=df, kind = "reg", color= "red")
    #st.pyplot()

    def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download csv file",
        data=csv,
        file_name="Download_df.csv",
        mime="text/csv",
    )

    if st.button("Exit"):
        st.balloons()



if __name__ == '__main__':
    main()


# In[ ]:




