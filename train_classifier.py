import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import re
import sys
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import GridSearchCV
import sqlite3
import pickle
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """Load data and return the data needed for the plot"""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('msg', engine)
    
    X=df['message']
    y=df.drop(['id','message','original','genre','related'],axis=1)
    return X, y, df

lemmatizer=WordNetLemmatizer()

def tokenize(text):
    """Tokenize words and remove stop words"""
    text=re.sub(r"[^A-Za-z0-9]"," ",text.lower())
    tokens=word_tokenize(text)
    tokens=[lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens
 
def build_model():
    """ Build a pipeline with a random forest as classifier, and update its parameters with the results obtained with GridSearch"""
    pipeline = Pipeline([
        
('vect', CountVectorizer(tokenizer=tokenize,ngram_range = (1, 2), max_df= 0.5, max_features = 5000)),
('tfidf', TfidfTransformer(use_idf= True, smooth_idf=True,sublinear_tf=True)),
('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators =40,min_samples_leaf=2, oob_score=True, random_state = None)))

    ])
  
    return pipeline
parameters = {
        'clf__estimator__n_estimators': [20,40], 
        'clf__estimator__min_samples_leaf':[2,4], 
        'vect__ngram_range': ((1, 1), (1, 2)), 
        'vect__max_df': (0.5, 1.0), 
        'vect__max_features': (None, 5000, 10000), 
        'tfidf__use_idf': (True, False), 
        'clf__estimator__oob_score' : (True, False),
        'clf__estimator__random_state':(None, 50),
         'tfidf__smooth_idf':( True,False),
        'tfidf__sublinear_tf': (False,True),
       'clf__estimator__criterion': ('gini','entropy') }           
search= GridSearchCV(build_model(), param_grid = parameters, n_jobs=-1, verbose=2)
search.fit(X_train,y_train)
print(search.best_params_)
print("Best parameter (CV score=%0.3f):" % search.best_score_)

def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate the model for each of the categories"""
    y_pred= model.predict(X_test)
    category_names = ['class 0', 'class 1']
    Y_test.index.name='index'
    Y_test.reset_index(inplace=True)
    Y_test=Y_test.drop(['index'],axis=1)
    
    y_pred=pd.DataFrame(y_pred)
    
    for i , row in enumerate (y_pred) :
        test=Y_test.loc[i]
        pred= y_pred.loc[i]
    return print(classification_report(test, pred))


def save_model(model, model_filepath):
    
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath =  sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
        
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