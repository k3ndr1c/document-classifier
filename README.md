# Document Classifier

A text document classifier for the Twenty Newsgroups dataset

## Installation

Install the necessary packages

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

Example workflow for a document classifier

```python
from DocumentClassifier import DocumentClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


feature = ('count-vec', CountVectorizer()
mode = ('naive-bayes', MultinomialNB())

document_classfier = DocumentClassifier()
document_classfier.load_data()
document_classfier.build_model(feature, model)
document_classfier.train_model()
document_classfier.evaluate_model()

feature_name = feature[0]
model_name = model[0]
f1_score = document_classfier.f1_score
print(f'{feature_name}, {model_name}, {f1_score}')
```

## License
[MIT](https://choosealicense.com/licenses/mit/)