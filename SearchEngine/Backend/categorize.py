import os
import pickle
import random
import re

import numpy as np
import pandas as pd  # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
# import Classification.main as main
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
regex = re.compile('[^a-zA-Z0-9 ]')


def get_words(headlines):
    headlines_onlyletters = regex.sub("", headlines)  # Remove everything other than letters
    words = headlines_onlyletters.lower().split()  # Convert to lower case, split into individual words
    meaningful_words = [wordnet_lemmatizer.lemmatize(word) for word in words if
                        word not in stop_words]  # Removing stopwords and lemmatize
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
    news["text"] = news["headline"] + news['short_description']
    news["category"] = news["category"].astype('category')
    news["category_code"] = news["category"].cat.codes
    news = news.reset_index(drop=True)
    dictionary = dict(enumerate(news["category"].cat.categories))
    save_dictionary(dictionary)
    index_for_validation = []
    for c in range(news["category_code"].nunique()):
        index = np.where(news["category_code"].values == c)[0]
        print(str(c) + "-" + str(len(index)))
        index_val = random.sample(list(index), 16)
        index_for_validation.extend(index_val)

    data_validate = news.iloc[index_for_validation]
    data_train = news.drop(index_for_validation)
    data_train = data_train.reset_index(drop=True)

    index_for_train = []
    for c in range(data_train["category_code"].nunique()):
        index = np.where(data_train["category_code"].values == c)[0]
        index_train = random.sample(list(index), 950)
        index_for_train.extend(index_train)
    data_train = data_train.iloc[index_for_train]
    data_train = data_train.reset_index(drop=True)
    text_train = list(data_train["text"])
    labels_train = list(data_train["category"])

    text_validate = list(data_validate["text"])
    labels_validate = list(data_validate["category"])

    # X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(news["headline"],
    #                                                                             news["category"],
    #                                                                             test_size=0.2)
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # Y_train = np.array(Y_train)
    # Y_test = np.array(Y_test)
    cleanHeadlines_train = []  # To append processed headlines
    cleanHeadlines_test = []  # To append processed headlines
    number_reviews_train = len(text_train)  # Calculating the number of reviews
    number_reviews_test = len(text_validate)  # Calculating the number of reviews
    for i in range(0, number_reviews_train):
        cleanHeadline = get_words(
            text_train[i])  # Processing the data and getting words with no special characters, numbers or html tags
        cleanHeadlines_train.append(cleanHeadline)
    for i in range(0, number_reviews_test):
        cleanHeadline = get_words(
            text_validate[i])  # Processing the data and getting words with no special characters, numbers or html tags
        cleanHeadlines_test.append(cleanHeadline)
    # Pipelined process
    # processing_step = Pipeline([('vect', CountVectorizer(analyzer="word")),
    #                             ('tfidf', TfidfTransformer()),
    #                             ('SVC', SVC())])
    # processing_step.fit(cleanHeadlines_train)

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

    # param_grid = [{'C': [1, 10, 100, 1000], 'tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10], 'max_iter': [1000, 2000, 3000, 4000, 5000]}]

    results = []
    names = []
    for name, model in models:
        # model_act = GridSearchCV(model, param_grid)
        # model_act.fit(X_train_tfidf, labels_train)
        # print(model_act.best_score_)
        # print(model_act.best_params_)
        # model_act.score(X_test_tfidf, labels_validate)
        model.fit(X_train_tfidf, labels_train)
        results.append(model.score(X_test_tfidf, labels_validate))
        names.append(name)
        # joblib.dump(model, 'Models/saved_model.pkl')
    for i in range(len(names)):
        print(names[i], results[i])
    joblib.dump(vectorize, "Models/saved_CounterVectorizer.pkl")
    joblib.dump(tfidf_transformer, "Models/saved_tfidf_transformer.pkl")
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
    vectorize = joblib.load(os.path.join("Models", "saved_CounterVectorizer.pkl"))
    tfidf_transformer = joblib.load(os.path.join("Models", "saved_tfidf_transformer.pkl"))
    dataset_title = []
    for doc_id in documents:
        dataset_title.append(get_words(doc_id.get_strings()))
    bagOfWords_test = vectorize.transform(dataset_title)
    test_tfidf = tfidf_transformer.transform(bagOfWords_test)
    predicted_category = load_model.predict(test_tfidf)
    for i, doc_id in enumerate(documents):
        doc_id.category = predicted_category[i]
    return documents
    # for i, title in enumerate(dataset_title):
    #     print(title + " ----- " + predicted_category[i])


def calculate_fscore(documents):
    index = np.arange(0, len(documents), 1)
    index_val = random.sample(list(index), 1000)
    categories = ['CRIME' 'ENTERTAINMENT' 'WORLD NEWS' 'POLITICS' 'COMEDY' 'SPORTS'
                  'BUSINESS' 'TRAVEL' 'MEDIA' 'TECH' 'RELIGION' 'SCIENCE' 'EDUCATION'
                  'ARTS & CULTURE' 'HEALTHY LIVING' 'WELLNESS' 'STYLE & BEAUTY'
                  'FOOD & DRINK' 'MONEY' 'ENVIRONMENT']
    predicted_value = []
    correct_value = []
    for ind in index_val:
        print("Title: " + documents[ind].title)
        print("Description: " + documents[ind].description)
        print("Category: " + documents[ind].category)
        for n, cat in enumerate(categories):
            print(str(n) + ": " + cat + " ")
            if cat == documents[ind].category:
                predicted_value.append(n)
        num = input("Please choose a category: ")
        correct_value.append(num)
    f1(correct_value, predicted_value)
    # con_matrix = confusion_matrix(correct_value, predicted_value)


def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1


def recall(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    print("Precision = " + str(p))
    print("Recall = " + str(r))
    if p + r == 0:
        print("F1 = 0")
    else:
        print("F1 = " + str(2 * (p * r) / (p + r)))



def save_dictionary(dictionary):
    with open('dictionary.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dictionary():
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Classification")
    with open(os.path.join(root_dir, 'dictionary.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# build_model()
