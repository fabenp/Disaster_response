import json
import plotly
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import plotly.graph_objs as go
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('msg', engine)
# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():  
    """  extract data needed for plotting twographs: one  with the percentage of all the categries from the total of related messages  and the         second with the percentage of each weather disaster compared to the total number of weather disasters"""
    df_weather=df[['floods','storm','earthquake','cold','other_weather']]
    df_weather=pd.DataFrame(df_weather.astype(int).sum(axis=0))
    df_weather['percentage']=(df_weather/(df_weather.sum(axis=1).sum()))
    percentage=df_weather['percentage'].values
    types=df_weather.index 
    # data needed to plot the percentages of the types of messages. All columns related to weather were dropped since the column' related to           weather" is the sum of them
    df_class=df.drop(['message','original','genre','earthquake','cold','other_weather','floods','storm'], axis=1)    
 
    list_positive_count=[]
    data_type=[]
    df_class=df_class[df_class['related']==1]
    for i in df_class.iloc[:,1:]:
        count_per_type=pd.DataFrame(df_class.groupby(i)['id'].count())
        try:
            positive_per_type=count_per_type.iloc[1,0]
            list_positive_count=[i,positive_per_type]
            data_type.append(list_positive_count)
        except:
            pass
    data_type=pd.DataFrame(data_type)
    data_type.columns=['message type','number']
    data_type['percentage']=data_type['number']/df_class.shape[0]
    data_type= data_type.sort_values('percentage',ascending=False)
    # create visuals   
    graphs=[
             dict(
            data= [dict(                          
                Bar(x=data_type['message type'],
                     y=data_type['percentage']))],
             layout=dict (
                title = 'Distribution of the related messages ',
                yaxis= {'title' : "Percentage"},                           
                xaxis= {'title': "Type"}  )),                                                                       
         dict(data= [dict(             
                Bar(x=types,
                    y=percentage))],                                              
            layout=dict (
                title = 'Weather emergencies',
                yaxis = {'title' : "Percentage"},                               
                xaxis = {'title': "Type of weather emergency"}))]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
