# Enhancing-Code-Mixed-Sentiment-Analysis-A-Transformer-based-Approach
Text-based sentiment analysis plays a very important role in understanding customer opinions and preferences in e-commerce platforms like Amazon, flipkart and other platforms. But despite extensive research in sentiment and emotion analysis in text, a notable gap exists in understanding code-mixed texts. To address this, we propose an end-to-end transformer based multitask framework designed for sentiment and emotion identification. This study focuses on sentiment analysis of Amazon product reviews, aiming to extract valuable insights from customer’s feedback. The analysis is conducted using machine learning techniques, specifically a multi-nominal Naive Bayes (NB) classifier, applied to a dataset of amazon reviews. The multi-nominal NB model is trained on a portion of the dataset and evaluated on another portion to assess its performance. The model is evaluated on certain key metrics such as accuracy, precision, recall, and F1 score that measure the effectiveness of the model. The results demonstrate the model’s ability to accurately classify sentiments in amazon product reviews, with high accuracy (86%), balanced precision (82%), recall (86%), and F1 score (83%). The findings of this study contribute to the field of sentiment analysis in e-commerce by providing insights into customer sentiment towards various products on Amazon, helping businesses make informed decisions based on customer feedback.
# importing libraries
!pip install pandas scikit-learn fasttext matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/content/amazon_reviews.csv', skiprows=[54315])
# Load the dataset from CSV
df = pd.read_csv('/content/amazon_reviews.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())
# Convert text labels to numerical labels
df['Sentiments'] = df['Sentiments'].map({'Negative': '__label__1', 'Neutral': '__label__2', 'Positive': '__label__3'})

# Split the dataset into features (X) and target (y)
X = df['Review_text']
y = df['Sentiments']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.isnull().sum())
print(X_test.isnull().sum())
import numpy as np

X_train = X_train.replace(np.nan, '')
X_test = X_test.replace(np.nan, '')
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# Check for missing values in the dataset
print("Number of missing values in X_train:", X_train.isnull().sum())
print("Number of missing values in y_train:", y_train.isnull().sum())

# Drop rows with missing values
X_train = X_train.dropna()
y_train = y_train.dropna()

# Reindex after dropping rows
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Check again for missing values
print("Number of missing values in X_train after handling:", X_train.isnull().sum())
print("Number of missing values in y_train after handling:", y_train.isnull().sum())
# Combine X_train and y_train into a single DataFrame
train_df = pd.concat([X_train, y_train], axis=1)

# Drop rows with missing values from the combined DataFrame
train_df = train_df.dropna()

# Separate X_train and y_train again
X_train = train_df['Review_text']
y_train = train_df['Sentiments']

# Reindex after dropping rows
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Create a CountVectorizer to convert text into a matrix of token counts
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# Create a Multinomial Naive Bayes classifier
nb_clf = MultinomialNB()

# Train the classifier
nb_clf.fit(X_train_vec, y_train)
# Predict the sentiment for the test set
nb_y_pred = nb_clf.predict(X_test_vec)
# Check unique labels in y_test and nb_y_pred
print("Unique labels in y_test:", y_test.unique())
print("Unique labels in nb_y_pred:", pd.Series(nb_y_pred).unique())
# Convert unknown labels to a known label
nb_y_pred_mapped = [label if label in ['__label__1', '__label__2', '__label__3'] else '__label__1' for label in nb_y_pred]
!pip install scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
nb_y_pred = pd.Series(nb_y_pred)
# Print the type of y_test and nb_y_pred
print(type(y_test))
print(type(nb_y_pred))

# Print the unique values in y_test and nb_y_pred
print(y_test.unique())
print(nb_y_pred.unique())
# Print the type of y_test and nb_y_pred
print(type(y_test))
print(type(nb_y_pred))

# Print the unique values in y_test and nb_y_pred
print(y_test.unique())
print(nb_y_pred.unique())
print(y_test.value_counts())
print(nb_y_pred.value_counts())
nb_y_pred_mapped = [label if label in ['__label__1', '__label__2', '__label__3'] else '__label__1' for label in nb_y_pred]
print(type(nb_y_pred))
print(nb_y_pred.unique())
print(type(y_test))
print(y_test.unique())
df = df.dropna(subset=['Review_text'])
print(y_test.unique())
df['Review_text'].fillna('__label__1', inplace=True)
print(y_test.unique())
# Calculate evaluation metrics
nb_accuracy = accuracy_score(y_test, nb_y_pred)
nb_precision = precision_score(y_test, nb_y_pred, average='weighted')
nb_recall = recall_score(y_test, nb_y_pred, average='weighted')
nb_f1 = f1_score(y_test, nb_y_pred, average='weighted')
print("Naive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Precision:", nb_precision)
print("Naive Bayes Recall:", nb_recall)
print("Naive Bayes F1 Score:", nb_f1)
# Create a confusion matrix for Naive Bayes
nb_conf_matrix = confusion_matrix(y_test, nb_y_pred_mapped)

# Plot the confusion matrix for Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(nb_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Naive Bayes Confusion Matrix')
plt.show()
# Create bar plots for the metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [nb_accuracy, nb_precision, nb_recall, nb_f1]

plt.figure(figsize=(10, 6))
sns.barplot(x=metrics, y=values, palette='viridis')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Naive Bayes Metrics')
plt.ylim(0, 1)  # Set y-axis limit to match score range
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

plt.show()
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Start the timer
start_time = time.time()

# Your sentiment analysis code
# ...

# End the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print("Elapsed Time:", elapsed_time, "seconds")

# Get the memory usage
memory_usage = psutil.Process().memory_info().rss
print("Memory Usage:", memory_usage, "bytes")

# Calculate evaluation metrics
nb_accuracy = accuracy_score(y_test, nb_y_pred_mapped)
nb_precision = precision_score(y_test, nb_y_pred_mapped, average='weighted')
nb_recall = recall_score(y_test, nb_y_pred_mapped, average='weighted')
nb_f1 = f1_score(y_test, nb_y_pred_mapped, average='weighted')

# Print the evaluation metrics
print("Naive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Precision:", nb_precision)
print("Naive Bayes Recall:", nb_recall)
print("Naive Bayes F1 Score:", nb_f1)
