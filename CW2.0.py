import pandas as pd
import nltk
import matplotlib.pyplot as plt
import string
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk.stem import PorterStemmer
ps = PorterStemmer()

data = pd.read_csv("C:/Users/fitz_/OneDrive - Edge Hill University/python/data analytics/movie_reviews.csv")
data.dropna(inplace=True)

print(data['sentiment'].value_counts())

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# cleaning the reviews
def clean(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = " ".join(ps.stem(word) for word in text.split() if word not in stop_words)
    
    words = text.split()
    words =[lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean)

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

sia = SentimentIntensityAnalyzer()

# Vader sentiment
def get_vader_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'pos'
    elif score <= -0.05:
        return 'neg'
    else:
        return 'neu'

data['vader_sentiment'] = data['clean_text'].apply(get_vader_sentiment)

print("VADER counts:")
print(data['vader_sentiment'].value_counts())

vader_data = data[data['vader_sentiment'] != 'neu']

true_vader = vader_data['sentiment'].map({'pos': 1, 'neg': 0})
pred_vader = vader_data['vader_sentiment'].map({'pos': 1, 'neg': 0})

vader_acc = accuracy_score(true_vader, pred_vader)
print("VADER Accuracy:", vader_acc)

X = data['clean_text']
y = data['sentiment'].map({'pos': 1, 'neg': 0})

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y, data['clean_text'], test_size=0.2, random_state=42
)

model = LinearSVC()
model.fit(X_train, y_train)

preds = model.predict(X_test)

svm_acc = accuracy_score(y_test, preds)
print("SVM Accuracy:", svm_acc)

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("SVM Confusion Matrix")
plt.show()

print("SVM predictions:")
print(pd.Series(preds).value_counts())

print("VADER predictions:")
print(data['vader_sentiment'].value_counts())

print("Comparison:")
print("SVM:", svm_acc)
print("VADER:", vader_acc)

results = pd.DataFrame({
    'review': text_test.values,
    'actual': y_test.values,
    'predicted': preds
})

wrong = results[results['actual'] != results['predicted']]

print("Wrong SVM predictions:")
print(wrong.head())

vader_results = pd.DataFrame({
    'review': vader_data['clean_text'].values,
    'actual': true_vader.values,
    'predicted': pred_vader.values
})

vader_wrong = vader_results[vader_results['actual'] != vader_results['predicted']]

print("Wrong VADER predictions:")
print(vader_wrong.head())

# test on new reviews

new_reviews = [
    "the movie was not bad at all",
    "i expected more from this film",
    "the plot was good but the acting was terrible",
    "it was okay nothing special",
    "i guess it was fine but not great",
    "this movie was so bad it was good",
    "i wouldn't say it's terrible but I wouldn't watch it again"]

# Clean them
new_reviews_clean = [clean(r) for r in new_reviews]

# Svm predictions
new_reviews_vec = tfidf.transform(new_reviews_clean)
svm_preds = model.predict(new_reviews_vec)

print("New review predictions (SVM):")
for i in range(len(new_reviews)):
    print("Review:", new_reviews[i])
    print("Prediction:", "Positive" if svm_preds[i] == 1 else "Negative")
    print()

# Vader predictions
print("New review predictions (VADER):")
for r in new_reviews:
    print("Review:", r)
    print("Prediction:", get_vader_sentiment(r))
    print()
