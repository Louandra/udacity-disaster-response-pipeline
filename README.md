# udacity-disaster-response-pipeline
## Introduction
This aim of this project is to create a classification model to identify categories in messages for disaster response purposes. 

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
- train_classifier.py: trains a multi class random forest classifier and saves the model
### app
- run.py: creates web app that allows the user to input a message and returns a classification result

## Setup
1. Run the process_data.py script
2. Run the train_classifier.py script
3. Run the run.py script
