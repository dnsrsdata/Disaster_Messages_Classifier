# Disaster Response Pipeline Project

## Instructions:
Follow the steps below to be able to play the project locally.

The Python version used in this project is 3.9.13. In order for the project to play smoothly, I suggest you are using the same version.
1. Clone the repository
   ```sh
   git clone https://github.com/dnsrsdata/Airbnb_Analysis.git
   ```
2. install the packages
   ```sh
   pip install -r requirements.txt
   ```
3. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```sh
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves
        ```sh
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run your web app.
    ```sh
    python run.py
    ```

3. Go to http://127.0.0.1:3000/

## Project Motivation
The idea for this project came from Udacity, where this is the second project proposed by Nanodegree. The project itself is a bit challenging, but I decided to tackle it because of the impact it would have, helping in critical disaster situations.

## File Descriptions
    - app
    | - template
    | |- master.html  # main page of web app
    | |- go.html  # classification result page of web app
    |- run.py  # Flask file that runs app
    |
    - data
    |- disaster_categories.csv  # data to process 
    |- disaster_messages.csv  # data to process
    |- process_data.py
    |- DisasterResponse.db   # database to save clean data 
    |
    - models
    |- train_classifier.py
    |- classifier.pkl  # saved model 
    |
    - README.md

## Results
You can check the result by accessing http://127.0.0.1:3000/ after following the installation steps

## Licensing, Authors, Acknowledgements
Must give credit to Appen for the data. Otherwise, feel free to use the code here as you would like!

