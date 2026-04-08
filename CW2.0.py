import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
import string
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer


# Load data
data = pd.read_csv(r"C:/Users/fitz_/OneDrive - Edge Hill University/python/data analytics/movie_reviews.csv")
data.dropna(inplace=True)

TfidfVectorizer(max_features=5000, ngram_range=(1,2))
max_features=5000 or 7000
stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_reviews(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = " ".join(ps.stem(word) for word in text.split() if word not in stop_words)
    
    # lemmatize + remove stopwords
    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )
    
    return text
data['text'] = data['text'].apply(clean_reviews)

# Negative word cloud (Red)
neg_reviews = ' '.join(
    data[data['sentiment'] == 'neg']['text'].astype(str)
)

wordcloud_neg = WordCloud(
    width=1000,
    height=600,
    colormap='Reds'
)

plt.figure(figsize=(15, 12))
plt.imshow(wordcloud_neg.generate(neg_reviews), interpolation='bilinear')
plt.title("Negative Reviews Word Cloud", fontsize=20)
plt.axis('off')
plt.show()

# Positive word cloud (Green)
pos_reviews = ' '.join(
    data[data['sentiment'] == 'pos']['text'].astype(str)
)

wordcloud_pos = WordCloud(
    width=1000,
    height=600,
    colormap='Greens'
)

plt.figure(figsize=(15, 12))
plt.imshow(wordcloud_pos.generate(pos_reviews), interpolation='bilinear')
plt.title("Positive Reviews Word Cloud", fontsize=20)
plt.axis('off')
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(max_features=2500)
reviews = cv.fit_transform(data['text']).toarray()

data['sentiment'] = data['sentiment'].map({'pos': 1, 'neg': 0})

print(data['sentiment'].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X = data['text']
y = data['sentiment']

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = {
    "Linear SVM": LinearSVC() }

for name, model in model.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, preds))
    
    
 





