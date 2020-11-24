from DocumentClassifier import DocumentClassifier

from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier


features = [
    ('count-vec', CountVectorizer()),
    ('tf-idf', TfidfVectorizer())
]
models = [
    ('log-reg', LogisticRegression(max_iter=100000)),
    ('naive-bayes', MultinomialNB()),
    ('linear-svm', LinearSVC(max_iter=100000)),
    ('xgb', XGBClassifier()),
]

results = []

document_classfier = DocumentClassifier()
document_classfier.load_data()

# Go through all the various feature to model pairs
for feature in features:
    for model in models:

            document_classfier.build_model(feature, model)
            document_classfier.train_model()
            document_classfier.evaluate_model()

            feature_name = feature[0]
            model_name = model[0]
            f1_score = document_classfier.f1_score
            results.append((feature_name, model_name, f1_score))


# Print out results to find best f1-score
for feature_name, model_name, f1_score in results:
    print(f'{feature_name}, {model_name}, {f1_score}')
