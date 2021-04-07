# AMLS_II_assignment20_21

This project was developed as an attempt of solution for SemEval-2017 Task 4: Twitter Sentiment Analysis. 
Specifically tasks A and B, Message polarity classification (based on a three-point scale) and Tweet classification 
according to a two-point scale respectively. The algorithm developed to solve both tasks was based on a specific type of 
Recurrent Neural Networks (RNN), Long Short-term Memory (LSTM). The classification is made on pre-processed text and a 
pre-trained word embedding. The performance of both tasks is measured using Average Recall Score, as it is in the 
official competition. Using this metric we obtained a score of 0.619 for task A and 0.798 for task B.

### Installation
Run the following commands to install the needed python libraries prior to run the code
```
$ pip3 install pandas
$ pip3 install numpy
$ pip3 install keras
$ pip3 install sklearn
$ pip3 install matplotlib
$ pip3 install tensorflow
$ pip3 install nltk
$ python3 -m nltk.downloader stopwords
```

### How to run the code
The code used for pre-processing the initial datasets should be commented out as the original files' location would fail.

Download the pre-trained word embedding from Glove at: https://drive.google.com/file/d/1x2u8tZSrG-plt7GT83_LHgqERxdtodwe/view?usp=sharing

Place the file on root directory

Run the main.py file to execute all four tasks, on root directory
```
$ python3 main.py
```


### Project Organisation
 ```
 .
 |-- main.py                      # Main file that will run the four tasks and call other Python files
 |-- A/
 |   |__ taskA.py                 # Message polarity classification train and test the model
 |-- B/
 |   |__ taskB.py                 # Tweet classification according to a two-point scale train and test the model
 |-- Datasets/                    # Folder containing the datasets used on two tasks
 |-- Models/                      # Folder containing different models pre-trained used to solve the tasks
 |-- Utils/
 |   |__ data_preprocessing.py    # Consists of functions used in different phases of feature engineering 
 |___
 ```
