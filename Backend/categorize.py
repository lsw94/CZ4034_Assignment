import re
import pandas as pd # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import sklearn
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score ,confusion_matrix
from sklearn.model_selection import cross_val_score


stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
regex = re.compile('[^a-zA-Z0-9 ]')


def get_words(headlines):
    headlines_onlyletters = regex.sub("", headlines)  # Remove everything other than letters
    words = headlines_onlyletters.lower().split()  # Convert to lower case, split into individual words
    meaningful_words = [wordnet_lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Removing stopwords and lemmatize
    return " ".join(meaningful_words)  # Joining the words


def build_model():
    news = pd.read_json("news-category-dataset/News_Category_Dataset_v2.json", lines=True)  # Importing data from CSV
    news = news.loc[(news['category'] != "WORLDPOST") & (news['category'] != "PARENTING") &
                    (news['category'] != "BLACK VOICES") & (news['category'] != "LATINO VOICES") &
                    (news['category'] != "QUEER VOICES") & (news['category'] != "WEIRD VOICES") &
                    (news['category'] != "IMPACT") & (news['category'] != "GOOD NEWS") &
                    (news['category'] != "TASTE") & (news['category'] != "HOME & LIVING") &
                    (news['category'] != "PARENTS") & (news['category'] != "PARENTING") &
                    (news['category'] != "WOMEN") & (news['category'] != "DIVORCE") &
                    (news['category'] != "WEIRD NEWS") & (news['category'] != "FIFTY") &
                    (news['category'] != "THE WORLDPOST") & (news['category'] != "GREEN") &
                    (news['category'] != "WEDDINGS") & (news['category'] != "ARTS") &
                    (news['category'] != "CULTURE & ARTS") & (news['category'] != "STYLE") &
                    (news['category'] != "COLLEGE")]
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
    # processing_step = Pipeline([('vect', CountVectorizer(analyzer="word")),
    #                             ('tfidf', TfidfTransformer()),
    #                             ('nb', MultinomialNB())])
    # processing_step.fit(X_train, Y_train)
    vectorize = CountVectorizer(analyzer="word")
    tfidf_transformer = TfidfTransformer()
    bagOfWords_train = vectorize.fit_transform(cleanHeadlines_train)
    X_train_tfidf = tfidf_transformer.fit_transform(bagOfWords_train)
    bagOfWords_test = vectorize.transform(cleanHeadlines_test)
    X_test_tfidf = tfidf_transformer.fit_transform(bagOfWords_test)

    # Use different model check accuracy metric
    models = []
    # models.append(("LogisticRegression", LogisticRegression()))
    # models.append(("SVC", SVC()))
    models.append(("LinearSVC", LinearSVC()))
    # models.append(("KNeighbors", KNeighborsClassifier()))
    # models.append(("DecisionTree", DecisionTreeClassifier()))
    # models.append(("MLPClassifier", MLPClassifier(solver='lbfgs', random_state=0)))
    # models.append(("MultinomialNB", MultinomialNB()))
    # models.append(("LogisticRegression", LogisticRegression()))

    results = []
    names = []
    for name, model in models:
        model.fit(X_train_tfidf, Y_train)
        results.append(model.score(X_test_tfidf, Y_test))
        names.append(name)
        joblib.dump(model, 'saved_model.pkl')
    for i in range(len(names)):
        print(names[i], results[i])
    # nb = MultinomialNB()
    # nb.fit(X_train_tfidf, Y_train)
    # print(nb.score(X_test_tfidf, Y_test))
    #
    # logistic_Regression = LogisticRegression()
    # logistic_Regression.fit(X_train_tfidf, Y_train)
    # Y_predict = logistic_Regression.predict(X_test_tfidf)
    # print(accuracy_score(Y_test, Y_predict))
    # joblib.dump(logistic_Regression, 'saved_model.pkl')


def categorize_document(documents):
    load_model = joblib.load(os.path.join("Models", "saved_model.pkl"))
    dataset_title = []
    for doc_id in documents:
        dataset_title.append(get_words(doc_id.title))
    vectorize = CountVectorizer(analyzer="word")
    tfidf_transformer = TfidfTransformer()
    bagOfWords_test = vectorize.fit_transform(dataset_title)
    test_tfidf = tfidf_transformer.fit_transform(bagOfWords_test)
    predicted_category = load_model.predict(test_tfidf)
    print(predicted_category)



build_model()

