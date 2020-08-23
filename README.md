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
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db  
python train_classifier.py  DisasterResponse.db classifier.pkl

to run the app,acces the folder containing the run.py file and type in  the command line:  
python run.py    
In another terminal window, run:  
env|grep WORK  
This will show the credentials needed for the web app address  

Link: https://github.com/fabenp/Disaster_response

![d](https://user-images.githubusercontent.com/55113875/90811669-01011580-e2f3-11ea-9a7c-c1dd4e4a68d3.GIF)
![d2](https://user-images.githubusercontent.com/55113875/90812343-fa26d280-e2f3-11ea-8691-cfc3cea7c4e1.GIF)
![d3](https://user-images.githubusercontent.com/55113875/90812357-001cb380-e2f4-11ea-81df-4eb3e5512bd7.GIF)
