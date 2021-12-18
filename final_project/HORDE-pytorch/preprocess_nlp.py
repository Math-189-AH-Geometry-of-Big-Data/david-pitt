# David Pitt
# Preprocess list of NLP concepts for HORDE model 

# Source for most code: https://github.com/Sue-Hi/NLP-MIMIC-III/blob/master/Word2Vec_Embedding_Mimic3.ipynb
# Adapted for my specific needs
data_dir = '/home/dave/Desktop/College/Math189AC/mimic3_1.4_files/'


import pandas as pd
import numpy as np

from nltk import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
from nltk.stem import *
from nltk.util import ngrams
import string
from nltk.corpus import stopwords
import re


from time import time

use_cols = ['SUBJECT_ID', 'HADM_ID', 'TEXT']
Notes = pd.read_csv(data_dir + "NOTEEVENTS.csv",usecols= use_cols, low_memory = False, engine = "c")

def preprocess_text(df):
    # This function preprocesses the text by filling not a number and replacing new lines ('\n') and carriage returns ('\r')
    df.TEXT = df.TEXT.fillna(' ')
    df.TEXT = df.TEXT.str.replace('\n',' ')
    df.TEXT = df.TEXT.str.replace('\r',' ')
    return df

Notes = preprocess_text(Notes)
print('Completed preprocessing text.')
Notes.to_csv('NOTEEVENTS_pre.csv')
def clean_text(text):
    punc_list = string.punctuation+'0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t).split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    # text = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)
    return text
print('Cleaning text...')
Notes['TEXT'] = Notes['TEXT'].map(lambda x: clean_text(x))

Notes.to_csv('NOTEEVENTS_NLP.csv')