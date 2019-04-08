import os
import random
import re

# import pickle
import numpy as np
import pandas as pd  # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

# import Classification.main as main

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
    main.save_dictionary(dictionary)
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
        dataset_title.append(get_words(doc_id.get_strings))
    bagOfWords_test = vectorize.transform(dataset_title)
    test_tfidf = tfidf_transformer.transform(bagOfWords_test)
    predicted_category = load_model.predict(test_tfidf)
    for i, doc_id in enumerate(documents):
        doc_id.category = predicted_category[i]
    # for i, title in enumerate(dataset_title):
    #     print(title + " ----- " + predicted_category[i])


def calculate_fscore(documents):
    index = np.arange(0, len(documents), 1)
    index_val = random.sample(list(index), 1000)
    categories = ['CRIME', 'ENTERTAINMENT', 'WORLD NEWS', 'POLITICS', 'COMEDY', 'SPORTS',
                  'BUSINESS', 'TRAVEL', 'MEDIA', 'TECH', 'RELIGION', 'SCIENCE', 'EDUCATION',
                  'ARTS & CULTURE', 'HEALTHY LIVING', 'WELLNESS', 'STYLE & BEAUTY',
                  'FOOD & DRINK', 'MONEY', 'ENVIRONMENT']
    predicted_value = []
    correct_value = []
    for m, ind in enumerate(index_val):
        print(m, end="")
        print("--------------------------------------------------------")
        print("Title: " + documents[ind].title)
        print("Description: " + documents[ind].description)
        print("Category: " + documents[ind].category)
        for n, cat in enumerate(categories):
            print(str(n) + ": " + cat)
            if cat == documents[ind].category:
                predicted_value.append(n)
        num = input("Please choose a category: ")
        while int(num) > 19 or int(num) < 0:
            num = input("Please choose a category: ")
        correct_value.append(int(num))
    print(predicted_value)
    print(correct_value)
    print(f1_score(correct_value, predicted_value, labels=list(np.arange(20)), average=None))
    print(f1_score(correct_value, predicted_value, labels=list(np.arange(20)), average='samples'))
    f1(correct_value, predicted_value)
    # con_matrix = confusion_matrix(correct_value, predicted_value)


def test():
    predicted_value = [3, 0, 8, 7, 8, 2, 18, 15, 2, 8, 12, 6, 2, 5, 6, 16, 3, 18, 8, 15, 3, 18, 2, 8, 18, 6, 0, 7, 5, 8,
                       5, 5, 8, 16, 13, 1, 8, 3, 10, 0, 5, 8, 9, 18, 19, 3, 3, 5, 4, 0, 8, 1, 3, 2, 2, 7, 19, 2, 2, 18,
                       8, 2, 13, 14, 19, 8, 5, 9, 13, 6, 5, 8, 0, 8, 2, 5, 3, 7, 2, 3, 19, 7, 7, 2, 19, 2, 9, 9, 4, 0,
                       2, 1, 15, 0, 0, 8, 12, 8, 9, 16, 9, 3, 8, 8, 8, 11, 6, 11, 9, 8, 2, 3, 5, 5, 0, 0, 3, 0, 19, 0,
                       7, 0, 3, 18, 1, 10, 2, 2, 8, 3, 3, 15, 6, 1, 2, 8, 6, 8, 8, 3, 2, 3, 8, 8, 3, 11, 9, 3, 12, 5, 9,
                       2, 5, 2, 8, 8, 5, 11, 14, 5, 2, 2, 9, 14, 9, 9, 0, 9, 4, 10, 2, 5, 9, 2, 5, 9, 2, 8, 8, 8, 8, 1,
                       18, 19, 19, 8, 8, 3, 11, 8, 13, 2, 3, 1, 7, 2, 8, 8, 19, 18, 3, 2, 13, 19, 9, 19, 0, 6, 8, 7, 8,
                       9, 19, 14, 18, 0, 3, 2, 0, 19, 9, 5, 10, 19, 7, 16, 8, 8, 17, 5, 13, 0, 5, 1, 2, 19, 4, 1, 8, 2,
                       11, 0, 18, 13, 14, 8, 13, 2, 1, 0, 3, 9, 9, 13, 3, 3, 18, 9, 7, 2, 5, 1, 3, 5, 8, 5, 3, 9, 8, 8,
                       5, 6, 14, 9, 0, 5, 3, 5, 5, 8, 8, 2, 10, 3, 14, 11, 3, 4, 2, 2, 8, 9, 5, 7, 8, 1, 16, 2, 19, 1,
                       8, 0, 2, 3, 5, 0, 19, 3, 13, 8, 8, 5, 8, 16, 0, 18, 5, 8, 2, 5, 2, 4, 8, 11, 5, 2, 3, 0, 18, 9,
                       8, 3, 1, 8, 5, 9, 16, 1, 8, 8, 11, 9, 8, 6, 8, 5, 8, 0, 8, 1, 11, 9, 1, 8, 12, 2, 8, 7, 2, 19, 3,
                       3, 8, 9, 6, 8, 9, 0, 5, 8, 6, 8, 9, 2, 11, 18, 5, 18, 8, 8, 0, 12, 5, 9, 18, 8, 11, 15, 19, 2,
                       18, 1, 8, 6, 15, 0, 2, 6, 3, 8, 5, 9, 1, 5, 3, 3, 4, 5, 15, 2, 1, 9, 6, 8, 8, 0, 8, 2, 5, 8, 11,
                       9, 0, 2, 5, 15, 0, 1, 9, 3, 2, 9, 8, 1, 10, 8, 5, 0, 5, 8, 3, 9, 3, 0, 13, 2, 8, 2, 18, 8, 3, 3,
                       8, 6, 0, 9, 2, 5, 8, 8, 3, 9, 3, 2, 0, 9, 8, 0, 8, 10, 7, 8, 2, 0, 17, 6, 5, 16, 2, 2, 5, 2, 6,
                       5, 3, 2, 2, 9, 3, 12, 8, 2, 16, 8, 7, 8, 9, 1, 17, 6, 8, 7, 3, 8, 2, 3, 2, 3, 8, 2, 3, 2, 2, 0,
                       12, 5, 3, 0, 5, 13, 2, 2, 8, 8, 0, 2, 0, 9, 9, 14, 5, 8, 6, 2, 0, 0, 5, 9, 5, 6, 7, 3, 9, 9, 8,
                       18, 9, 8, 1, 4, 11, 8, 16, 18, 0, 3, 8, 9, 5, 8, 12, 2, 6, 0, 2, 3, 9, 11, 2, 8, 15, 1, 3, 0,
                       3, 0, 2, 3, 9, 0, 1, 8, 8, 1, 9, 8, 6, 2, 1, 8, 8, 12, 0, 9, 2, 2, 14, 1, 2, 12, 8, 11, 17, 8, 3,
                       8, 2, 7, 11, 2, 1, 6, 3, 19, 5, 8, 8, 8, 8, 5, 9, 8, 8, 2, 4, 0, 5, 2, 2, 3, 12, 8, 8, 3, 2, 2,
                       0, 5, 6, 13, 3, 0, 19, 3, 5, 8, 17, 3, 3, 17, 9, 9, 0, 8, 3, 14, 18, 0, 2, 6, 8, 2, 11, 2, 3, 8,
                       6, 8, 8, 2, 2, 8, 15, 8, 7, 8, 2, 15, 0, 0, 5, 19, 16, 6, 12, 8, 5, 2, 8, 3, 2, 3, 18, 2, 2, 6,
                       9, 5, 0, 3, 8, 8, 0, 5, 13, 2, 16, 9, 2, 3, 2, 8, 18, 6, 9, 3, 0, 19, 2, 2, 0, 8, 6, 1, 6, 12, 2,
                       5, 0, 6, 9, 2, 5, 3, 3, 0, 8, 7, 2, 3, 2, 8, 6, 2, 8, 19, 3, 8, 12, 9, 0, 9, 6, 1, 8, 8, 3, 8, 2,
                       8, 8, 6, 8, 0, 8, 7, 8, 5, 8, 2, 11, 5, 6, 8, 7, 11, 2, 16, 0, 1, 8, 9, 5, 15, 5, 17, 11, 0, 2,
                       16, 8, 17, 16, 2, 8, 8, 3, 19, 7, 14, 6, 7, 3, 7, 12, 0, 8, 3, 2, 1, 15, 2, 9, 9, 3, 7, 14, 1, 8,
                       8, 2, 16, 8, 1, 3, 11, 8, 2, 7, 10, 2, 3, 2, 2, 1, 3, 18, 12, 5, 0, 3, 9, 3, 8, 9, 3, 2, 3, 8, 3,
                       3, 0, 16, 2, 0, 6, 8, 3, 2, 15, 8, 16, 8, 2, 2, 14, 9, 15, 6, 6, 12, 0, 3, 8, 8, 7, 8, 8, 8, 9,
                       3, 8, 8, 3, 8, 6, 0, 2, 19, 8, 8, 0, 3, 18, 3, 3, 3, 2, 2, 8, 8, 4, 8, 2, 2, 3, 18, 2, 0, 4, 9,
                       0, 2, 5, 7, 2, 9, 5, 3, 2, 16, 3, 9, 0, 2, 3, 2, 8, 8, 5, 7, 3, 0, 15, 1, 17, 3, 8, 2, 3, 3, 0,
                       8, 5, 3, 2, 3, 8, 15, 0, 8, 9, 5, 6, 19, 7, 5, 1, 1, 8, 9, 10, 19, 3, 8, 0, 10, 2, 14, 5, 8, 6,
                       11, 18, 19, 6, 11, 9, 3, 2, 6, 3, 19, 5, 16, 8, 7, 3, 8, 3, 5, 5, 3, 8, 0, 3, 8, 2, 8, 8]

    correct_value = [3, 0, 8, 1, 3, 8, 1, 15, 6, 8, 8, 6, 2, 5, 3, 16, 3, 18, 8, 8, 3, 18, 18, 8, 11, 6, 8, 7, 5, 8, 5,
                     5, 1, 8, 13, 8,
                     8, 3, 3, 0, 5, 8, 9, 8, 8, 3, 3, 5, 4, 0, 8, 1, 3, 2, 2, 8, 9, 2, 2, 18, 8, 2, 13, 14, 19, 8, 5, 9,
                     12, 6, 5, 8, 0,
                     8, 2, 5, 3, 6, 2, 3, 8, 8, 7, 2, 19, 8, 9, 9, 1, 0, 18, 1, 5, 0, 8, 8, 12, 8, 9, 16, 9, 3, 3, 8, 8,
                     11, 6, 11, 9,
                     8, 2, 3, 8, 5, 8, 0, 3, 0, 8, 8, 7, 8, 3, 18, 1, 10, 2, 2, 8, 3, 3, 15, 6, 1, 2, 8, 6, 8, 8, 3, 8,
                     3, 8, 8, 3, 11,
                     9, 3, 12, 5, 9, 2, 5, 2, 8, 8, 5, 11, 14, 5, 8, 2, 8, 14, 9, 12, 0, 9, 5, 5, 2, 5, 9, 2, 5, 8, 2,
                     8, 8, 8, 8, 1, 2,
                     8, 6, 3, 8, 8, 11, 8, 1, 2, 3, 1, 1, 2, 8, 8, 8, 18, 3, 2, 8, 8, 8, 8, 8, 6, 8, 7, 8, 9, 19, 14,
                     18, 5, 3, 2, 8,
                     19, 9, 5, 10, 19, 7, 8, 19, 6, 17, 5, 8, 8, 5, 1, 2, 19, 1, 1, 8, 3, 11, 19, 18, 13, 14, 2, 16, 2,
                     8, 0, 8, 9, 9,
                     8, 3, 3, 18, 9, 19, 2, 5, 8, 3, 8, 8, 5, 3, 9, 8, 8, 5, 6, 14, 9, 0, 5, 3, 5, 5, 8, 8, 2, 12, 3,
                     14, 19, 3, 4, 2,
                     2, 8, 9, 5, 7, 16, 1, 1, 2, 19, 8, 8, 8, 2, 3, 5, 0, 5, 3, 8, 3, 8, 1, 8, 16, 0, 18, 5, 8, 2, 5, 2,
                     8, 8, 1, 5, 2,
                     3, 0, 8, 9, 8, 3, 1, 8, 5, 9, 11, 1, 8, 15, 8, 9, 16, 3, 8, 8, 6, 0, 8, 1, 11, 9, 1, 8, 12, 2, 8,
                     8, 2, 19, 3, 3,
                     8, 9, 8, 8, 9, 8, 5, 3, 6, 8, 9, 2, 11, 18, 5, 8, 8, 12, 8, 12, 5, 9, 18, 8, 11, 1, 2, 2, 8, 1, 7,
                     6, 15, 2, 2, 15,
                     8, 8, 5, 9, 1, 5, 3, 3, 0, 5, 8, 2, 5, 9, 6, 8, 8, 0, 8, 2, 5, 1, 11, 9, 0, 2, 5, 15, 0, 1, 8, 3,
                     2, 8, 8, 1, 2, 8,
                     5, 0, 5, 8, 2, 9, 3, 2, 1, 2, 1, 2, 18, 8, 3, 3, 8, 6, 0, 9, 2, 5, 8, 8, 8, 9, 3, 2, 0, 9, 17, 8,
                     8, 10, 7, 8, 2,
                     8, 17, 6, 5, 8, 2, 2, 5, 2, 6, 5, 3, 2, 2, 8, 3, 3, 8, 2, 9, 8, 7, 19, 9, 1, 19, 6, 8, 8, 3, 1, 2,
                     3, 2, 3, 3, 2,
                     3, 2, 2, 0, 12, 5, 3, 0, 5, 13, 2, 2, 8, 8, 8, 2, 0, 9, 9, 14, 5, 8, 6, 2, 0, 0, 5, 9, 5, 6, 8, 3,
                     9, 9, 8, 18, 9,
                     3, 1, 8, 9, 8, 16, 18, 0, 3, 8, 9, 5, 8, 12, 2, 6, 0, 8, 3, 9, 8, 2, 8, 15, 1, 3, 0, 3, 0, 2, 3, 9,
                     0, 1, 8, 8, 1,
                     9, 8, 8, 2, 1, 8, 15, 12, 0, 9, 2, 2, 0, 1, 2, 12, 7, 11, 9, 8, 3, 8, 2, 7, 1, 2, 1, 6, 3, 8, 5, 8,
                     8, 11, 8, 5, 9,
                     8, 1, 2, 8, 0, 5, 2, 2, 3, 12, 8, 8, 1, 2, 2, 8, 5, 8, 13, 3, 8, 8, 3, 5, 8, 17, 3, 3, 11, 9, 9, 0,
                     8, 8, 14, 18,
                     0, 2, 8, 14, 2, 11, 2, 3, 16, 6, 8, 8, 2, 2, 8, 1, 8, 7, 8, 5, 8, 9, 3, 8, 19, 16, 6, 8, 8, 5, 2,
                     6, 3, 5, 3, 8, 2,
                     3, 6, 9, 5, 0, 3, 8, 8, 8, 5, 5, 2, 16, 9, 2, 3, 2, 8, 18, 6, 9, 19, 0, 19, 2, 2, 0, 8, 6, 1, 6, 8,
                     2, 5, 0, 6, 9,
                     2, 5, 5, 3, 0, 8, 8, 2, 3, 2, 13, 5, 1, 8, 19, 3, 8, 12, 9, 8, 9, 8, 8, 2, 8, 3, 8, 2, 8, 8, 6, 8,
                     8, 8, 7, 8, 5,
                     8, 2, 8, 5, 6, 8, 7, 11, 2, 19, 0, 8, 8, 9, 5, 15, 5, 17, 11, 2, 2, 16, 8, 17, 16, 2, 8, 8, 8, 17,
                     1, 14, 8, 1, 8,
                     2, 12, 0, 8, 3, 5, 8, 15, 2, 9, 9, 3, 8, 14, 1, 8, 3, 2, 8, 8, 8, 3, 11, 8, 2, 6, 8, 8, 3, 2, 2, 1,
                     6, 18, 12, 5,
                     0, 3, 5, 3, 8, 9, 8, 2, 8, 8, 3, 3, 0, 16, 6, 2, 8, 8, 3, 2, 15, 8, 8, 8, 2, 2, 8, 9, 15, 6, 6, 12,
                     8, 3, 8, 8, 7,
                     8, 8, 8, 9, 3, 8, 2, 3, 8, 6, 8, 2, 19, 8, 8, 0, 3, 1, 8, 3, 3, 5, 2, 8, 8, 4, 8, 2, 2, 8, 18, 2,
                     8, 9, 9, 0, 2, 5,
                     7, 2, 9, 5, 3, 2, 16, 3, 8, 0, 2, 3, 2, 8, 8, 5, 8, 3, 8, 15, 1, 17, 3, 8, 8, 3, 3, 0, 8, 5, 3, 2,
                     3, 8, 15, 0, 6,
                     8, 8, 6, 19, 8, 5, 8, 5, 8, 9, 2, 13, 8, 8, 8, 10, 2, 14, 5, 8, 8, 11, 18, 8, 3, 11, 9, 3, 2, 6, 3,
                     19, 5, 16, 8,
                     8, 3, 8, 8, 5, 5, 3, 8, 0, 3, 8, 2, 8, 8]

    print(predicted_value)
    print(correct_value)
    print(f1_score(correct_value, predicted_value, labels=list(np.arange(20)), average=None))
    print(f1_score(correct_value, predicted_value, labels=list(np.arange(20)), average='micro'))
    print(f1_score(correct_value, predicted_value, labels=list(np.arange(20)), average='weighted'))
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

test()