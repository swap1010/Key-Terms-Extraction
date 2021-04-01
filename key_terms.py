import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import string

with open('news.xml', 'r') as f:
    data = f.read()
soup = BeautifulSoup(data, "xml")
news = soup.find_all("news")
titles = []
punctuations = list(string.punctuation)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
headlines = []
matter = []
for new in news:
    head, tail = new.find_all("value")
    tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(tail.text.lower())]
    tags = [nltk.pos_tag([token])[0] for token in tokens if token not in stop_words and token not in punctuations]
    nouns = [word for word, tag in tags if tag == "NN"]
    matter.append(" ".join(nouns))
    headlines.append(head.text + ":")

vector = vectorizer.fit_transform(matter)
words = vectorizer.get_feature_names()
for i in range(len(headlines)):
    tfidf_metric = vector.toarray()[i]
    res = sorted(range(len(tfidf_metric)), key=lambda sub: (tfidf_metric[sub], words[sub]), reverse=True)[:5]
    print(headlines[i])
    text = []
    for ind in res:
        text.append(words[ind])
    print(" ".join(text))
