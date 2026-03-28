import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download('stopwords')

# Load data
data = pd.read_csv(r"C:/Users/fitz_/OneDrive - Edge Hill University/python/data analytics/movie_reviews.csv")
data.dropna(inplace=True)

stop_words = set(stopwords.words('english'))

# Clean reviews
def clean_reviews(review):
    return ' '.join(
        word for word in review.split()
        if word.lower() not in stop_words
    )

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
