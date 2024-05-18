import sys


def load_data(database_filepath):
    """Loads data from database"""
    import pandas as pd
    from sqlalchemy import create_engine
    
    # load data from database
    # engine = create_engine('sqlite:///la_udacity_project.db')
    engine = create_engine(database_filepath)
    query = "SELECT * FROM disaster_response_data"
    df = pd.read_sql_query(query, engine)

    X = df['message'].values
    Y = df.drop(columns = ['id', 'message', 'original', 'genre']).values

    category_names = list(df.drop(columns = ['id', 'message', 'original', 'genre']).columns)

    return X, Y, category_names

def tokenize(text):
    """
    Splits text into tokens
    
    Arguments:
    text (str): text to split
    Returns:
    tokens (list): list of tokens in the text
    
    """
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize 
    from nltk.stem.wordnet import WordNetLemmatizer

    stop_words = stopwords.words("english")
    # convert text to lowercase
    text = text.lower()
    
    # remove punctuation
    import re
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    words = word_tokenize(text) 

    words = [w for w in words if w not in stop_words] 
    
    lemmatizer = WordNetLemmatizer()    
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return tokens


def build_model(X_train, y_train):
    """
    Trains a text classifier model and returns the model object
    """
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier

    # train pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer(smooth_idf=False))
            ])),
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # grid search parameters
    from sklearn.model_selection import GridSearchCV

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 150],
        'clf__estimator__min_samples_split': [5, 10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 5)
    cv.fit(X_train, y_train)

    best_estimator = cv.best_estimator_
    # get best model from gridsearch
    model = best_estimator

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Calculates accuracy metrics for each category
    Args:
    - model: trained model
    - X_test: Text
    - Y_test: Labels
    """
    from sklearn.metrics import accuracy_score, classification_report

    # predict on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    
    # accuracy for each class
    accuracies = [accuracy_score(Y_test[:, i], y_pred[:, i]) for i in range(Y_test.shape[1])]
    print("Accuracy for each output:", accuracies)


def save_model(model, model_filepath):
    import joblib
    joblib.dump(model, f'{model_filepath}.pkl')

def main():
    from sklearn.model_selection import train_test_split
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()