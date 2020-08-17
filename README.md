## Summary
The  disaster  response web app classifies messages by category in order to filter them in case of a disaster and enables a better response. To achieve this result, a pipeline with a random forest classifier was used to work on the training data. 

## Libraries used
numpy, pandas, nltk
## Files in this repository
### Training data:
disaster_categories.csv: A table with the id of the messages and their categories. There's often more than one category associated with a message.  
disaster_messages.csv: A table with the original message (many are in another language), the translated message, its id  and its  genre (social, news, direct).
### HTML files:
go.html : HTML code needed to return the classification result of the  message.  
master.html: HTML code for the design of the web app.
### Cleaning data:
process_data.py: prepare the training data and store it for later use .
### Training file: 
train_classifier.py: use the cleaned data to train a pipeline and improve the accuracy of prediction of the model
### Run the  web app:
run.py: file to run in order to access the web app. It contains the data needed for the plots and the code to plot the barplots. 
## Run the python script:
run in the command line:
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db  
python models/train_classifier.py  data/DisasterResponse.db models/classifier.pkl

to run the app,acces the folder containing the run.py file and type in  the command line:  
python run.py    
In another terminal window, run:  
env|grep WORK  
This will show the credentials needed for the web app address  
