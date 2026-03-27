import pandas as pd
import nltk
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

texts = [" ".join(words) for words, sentiment in documents]
labels = [sentiment for words, sentiment in documents]

nltk.download('movie_reviews')
data = pd.DataFrame({'text': texts, 'sentiment': labels})
data.to_csv('movie_reviews.csv', index=False)
print("Movie reviews dataset saved to 'movie_reviews.csv'.")





import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset from the CSV file
data = pd.read_csv('movie_reviews.csv')
texts = data['text'].tolist()
labels = data['sentiment'].tolist()


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print ('Training the AI model...')
classifier = SVC(kernel='linear')
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

new_review = ["This movie was great!"]
new_review_vec = vectorizer.transform(new_review)
prediction = classifier.predict(new_review_vec)
print(f"Predicted sentiment for the new review: {prediction[0]}")



import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Use TF-IDF instead of CountVectorizer
from sklearn.svm import LinearSVC  # Use LinearSVC for better performance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

# Prepare the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

data = pd.read_csv('movie_reviews.csv')
texts = data['text'].tolist()
labels = data['sentiment'].tolist()

# Text preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to the texts
texts = [preprocess_text(text) for text in texts]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Use TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to avoid overfitting
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 

# Train a LinearSVC classifier
classifier = LinearSVC(C=1.0, random_state=42)  # Adjust C for regularization
classifier.fit(X_train_vec, y_train)

# Predict the sentiment on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




# Example of predicting sentiment for a new review
new_review = ["good movie"]
new_review = [preprocess_text(new_review[0])]  # Preprocess the new review
new_review_vec = vectorizer.transform(new_review)
prediction = classifier.predict(new_review_vec)
print(f"\nPredicted sentiment for the new review: {prediction[0]}")





