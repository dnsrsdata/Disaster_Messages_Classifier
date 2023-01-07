import sys
import pandas as pd
import re
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import warnings

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

warnings.simplefilter('ignore')


def load_data(database_filepath):
    """
    Load data from an SQLite database
    :param database_filepath: database name
    :return: explanatory features, target and an array with dataframe columns
    """
    # Creating a connection and reading the data
    engine = create_engine(f'sqlite:///{database_filepath}')
    df_disaster_categories = pd.read_sql_table('disaster_categories', engine)

    # Splitting the data
    x = df_disaster_categories.message
    y = df_disaster_categories.iloc[:, 2:]
    y = y.drop(columns=y.columns[y.sum() == 0])

    return x, y, y.columns


def tokenize(text):
    """
    Transforms the set of texts into vectors
    :param text: text/phrase/sentence to be tokenized
    :return: tokenized text
    """
    stopwords_list = stopwords.words('english')

    # Remove punctuation and urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    new_text = re.sub(url_regex, ' ', text)
    new_text = re.sub('\W', ' ', new_text)

    # Removing stopwords, bringing words to their root form and normalize them
    tokens = word_tokenize(new_text)
    lemmatizer = WordNetLemmatizer()
    new_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stopwords_list]

    return new_tokens


def build_model():
    """
    Build a Decision Tree model
    :return: model
    """
    # Building a pipeline
    pipe = Pipeline([('vectorizer', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', DecisionTreeClassifier())])

    # Setting the parameters
    params = {'clf__max_depth': [6, 15],
              'clf__min_samples_split': [20],
              'clf__criterion': ['gini']}

    # Looking for the best combination
    grid_clf = GridSearchCV(pipe, params, cv=5)

    return grid_clf


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate model metrics like precision, recall and f1-score
    :param model: machine learning model
    :param X_test: test set with explanatory features
    :param Y_test: test set with target features
    :param category_names: name of columns
    :return: None
    """
    # Checking the metrics
    y_predict = model.predict(X_test)
    print(classification_report(Y_test, y_predict, target_names=category_names))


def save_model(model, model_filepath):
    """
    save the model to a .pkl file
    :param model: trained model
    :param model_filepath: name to .pkl file
    :return: None
    """
    joblib.dump(model, f'{model_filepath}')


def main():
    """
    run all above steps
    :return:
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
