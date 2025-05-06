import pandas as pd
import streamlit as st
import cleantext
import torch
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_sia():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model


st.header('🔎 WACAPUBLIKA')
st.markdown('Your go-to public sentiment analyzer. Instantly discover the emotion behind any text—just type it in and get insights!✨<br>'
            '<span style="color:gray"><i>Note: Only English text is supported at the moment.</i></span>', unsafe_allow_html=True)
engine = st.radio("Choose Sentiment Engine:", ['TextBlob Engine', 'SIA Engine', 'BERT Engine', 'LSTM Engine', 'Naive Bayes Engine'], horizontal=True)

with st.expander('Analyze Text'):  
    text = st.text_input('Text here')

    def sentiment(score):
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return'Neutral'

    if text:
        if engine == 'TextBlob Engine':
            blob = TextBlob(text)
            st.write('Polarity: ', round(blob.sentiment.polarity, 2))
            st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))
            st.write('Sentiment : ', sentiment(blob.sentiment.polarity))
        elif engine == 'SIA Engine' :
            sia = load_sia()
            sia = sia.polarity_scores(str(text))['compound']
            st.write('Polarity: ', round(sia, 2))
            st.write('Sentiment : ', sentiment(sia))
        else:
            tokenizer, model = load_bert()
            token = tokenizer.encode(text, return_tensors = 'pt')
            with torch.no_grad():
                result = model(token)
            score = int(torch.argmax(result.logits)) + 1

            if score > 3:
                output = 'Positive'
            elif score < 3:
                output = 'Negative'
            else:
                output = 'Neutral'

            st.write('Polarity: ', score)
            st.write('Sentiment : ', output)

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 stopwords=True, lowercase=True, numbers=True, punct=True))

with st.expander('Analyze File'): 
    upl = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx'])

    def score_blob(text):
        blob1 = TextBlob(text)
        return blob1.sentiment.polarity  
         
    def score_sia(text):
        sia = load_sia()
        return sia.polarity_scores(str(text))['compound']

    def analyze(score):
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def analyze_bert(text):
        tokenizer, model = load_bert()
        token = tokenizer.encode(text, return_tensors = 'pt')
        with torch.no_grad():
            result = model(token)
        score = int(torch.argmax(result.logits)) + 1
        if score > 3:
            output = 'Positive'
        elif score < 3:
            output = 'Negative'
        else:
            output = 'Neutral'
        return score, output

    if upl is not None:
        try:
            if upl.name.endswith('.csv'):
                df = pd.read_csv(upl, delimiter=';')
            elif upl.name.endswith('.xlsx'):
                df = pd.read_excel(upl)
            else:
                st.warning("Unsupported file format. Please upload a CSV or Excel file.")
                st.stop()

            if 'Unnamed: 0' in df.columns and len(df.columns) > 1:
                del df['Unnamed: 0']

            if 'comments' not in df.columns:
                st.error("The uploaded file must contain a 'comments' column.")
                st.stop()

            if engine == 'TextBlob Engine':
                df['score'] = df['comments'].apply(score_blob)
                df['analysis'] = df['score'].apply(analyze)
            elif engine == 'SIA Engine':
                df['score'] = df['comments'].apply(score_sia)
                df['analysis'] = df['score'].apply(analyze)
            else:
                df[['score', 'analysis']] = df['comments'].apply(lambda x: pd.Series(analyze_bert(x)))

            st.write(df.head(10))

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")
