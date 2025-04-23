import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load data
df = pd.read_csv("C:/FUTURE_DS_02/notebook/data/customer_support_tickets.csv")

# Handle missing values
df.dropna(subset=['Ticket Description','First Response Time','Time to Resolution','Ticket Type'], inplace=True)

# Format datetime columns
df['First Response Time'] = pd.to_datetime(df['First Response Time'])
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'])

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b(product\s?purchased|i bought|ordered|your product|please|assist|im)\b','', text, flags=re.IGNORECASE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return re.sub(r'\s+', ' ', " ".join(tokens)).strip()

# Apply cleaning
df['clean_description'] = df['Ticket Description'].apply(clean_text)

# Remove common phrases
COMMON_PHRASES = [
    'im issue', 'ive noticed', 'issue persists', 'im unable', 'issue im', 'ive tried',
    'im facing', 'resolve problem', 'ive checked', 'troubleshooting steps',
    'ive recently', 'guide steps', 'unable option', 'option perform', 'perform desired',
    'desired action', 'action guide', 'thank', 'product', 'find', 'im using', 'occurring',
    'thanks', 'started', 'recent','item','didnt','started','havent','happening','help',
    'issue','afterward','ive followed','soon','possible','peculiar','says','mean','works fine',
    'ive already','problem','ive performed','multiple times','times remain','order'
]

def remove_common_phrases(text):
    for phrase in COMMON_PHRASES:
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df['clean_description'] = df['clean_description'].apply(remove_common_phrases)

# POS tag mapping
from nltk import pos_tag, word_tokenize

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize_with_pos(text):
    tokens = word_tokenize(str(text))
    tagged_tokens = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
    return " ".join(lemmatized)

df['lemmatized_description'] = df['clean_description'].apply(lemmatize_with_pos)

# Save cleaned dataset
df.to_csv('notebook/data/lemmatized_tickets.csv', index=False)

# Final check
print("Preprocessing complete. Sample:")
print(df[['Ticket Description','clean_description','lemmatized_description','First Response Time','Time to Resolution','Ticket Type']].head())
print(df.info())
print(df.columns)

# Train-test split for future modeling
X_train, X_test, y_train, y_test = train_test_split(df['clean_description'], df['Ticket Description'], test_size=0.2, random_state=42)
