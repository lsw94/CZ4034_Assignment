import re
import pandas as pd # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix


stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
regex = re.compile('[^a-zA-Z0-9 ]')


def get_words( headlines ):
    headlines_onlyletters = regex.sub("", headlines)  # Remove everything other than letters
    words = headlines_onlyletters.lower().split()  # Convert to lower case, split into individual words
    meaningful_words = [wordnet_lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Removing stopwords and lemmatize
    return " ".join(meaningful_words)  # Joining the words


news = pd.read_json("news-category-dataset/News_Category_Dataset_v2.json", lines=True)  # Importing data from CSV
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(news["headline"],
                                                                            news["category"],
                                                                            test_size=0.2)
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
cleanHeadlines_train = [] #To append processed headlines
cleanHeadlines_test = [] #To append processed headlines
number_reviews_train = len(X_train) #Calculating the number of reviews
number_reviews_test = len(X_test) #Calculating the number of reviews
for i in range(0,number_reviews_train):
    cleanHeadline = get_words(X_train[i]) #Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadlines_train.append(cleanHeadline)
for i in range(0,number_reviews_test):
    cleanHeadline = get_words(X_test[i]) #Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadlines_test.append(cleanHeadline)
vectorize = sklearn.feature_extraction.text.CountVectorizer(analyzer="word", max_features=10000)
bagOfWords_train = vectorize.fit_transform(cleanHeadlines_train)
X_train = bagOfWords_train.toarray()
bagOfWords_test = vectorize.transform(cleanHeadlines_test)
X_test = bagOfWords_test.toarray()

vocab = vectorize.get_feature_names()
nb = MultinomialNB()
nb.fit(X_train, Y_train)
print(nb.score(X_test, Y_test))

logistic_Regression = LogisticRegression()
logistic_Regression.fit(X_train,Y_train)
Y_predict = logistic_Regression.predict(X_test)
print(accuracy_score(Y_test,Y_predict))
joblib.dump(nb, 'saved_model.pkl') 

