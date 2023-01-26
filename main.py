import streamlit as st
import joblib

model_file = open('Hate Speech Classifier.joblib', 'rb')
model = joblib.load(model_file)

st.write("""
# Cek Hate Speech Tweet KDRT
""")

data = st.text_input('Masukkan Teks')
cek = st.button('Cek Prediksi')

input_tweet = [data]

def preProcessText(tweer):
    new_tweets = []
    for tw in texts:
        tw = case_folding(tw)
        tw = tokenized(tw)
        tw = stemming(tw)
        tw = removeSlang(tw)
        tw = removeStopWords(tw)
        tw = ' '.join(tw)
        new_tweets.append(tw)

    return new_tweets

def predictNewData(tweets):
    saved_model = joblib.load('Hate Speech Classifier.joblib') 
    saved_tfidf = joblib.load('Hate Speech TF-IDF Vectorizer.joblib')

    vectorized_tweets = saved_tfidf.transform(tweets)
    input_prediction = saved_model.predict(vectorized_tweets)

    for i in range(len(input_tweet)):
        if cek :
            if input_prediction[i]==1:
                st.write('Input text: ', 
                input_tweet[i]) 
                st.write('Prediction: Hate Speech!')
            else:
                st.write('Input text: ', 
                input_tweet[i]) 
                st.write('Prediction: Not a Hate Speech.')
            
predictNewData(input_tweet)