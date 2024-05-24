# udacity-disaster-response-pipeline
## Introduction
This aim of this project is to create a classification model to identify categories in messages for disaster response purposes. The output is a front end user interface which allows a user to type in a message and view a list of categories the text falls within.

## Technologies
- Python
- HTML/CSS
- Bootstrap
- Plotly
- Flask

## Components
### data
- disaster_messages.csv: contains the message ID, original message, translated message and the genre
- disaster_categories.csv: contains the categories the message can be classified into
- process_data.py: this script creates a SQLLite database containing the training data created from disaster_messages.csv and disaster_categories.csv
### models 
- train_classifier.py: trains a multi class random forest classifier and saves the model. GridSearchCV was used to find the optimal parameters.
### app
- run.py: createsa web app that allows the user to input a message and returns a classification result

## Set up and execution
- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
- Run the following command in the app's directory to run the web app. python run.py
- Go to http://0.0.0.0:3001/

## Future Work
1. Model tuning
2. Web app refinement

