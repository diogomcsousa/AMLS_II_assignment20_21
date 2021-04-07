import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import numpy as np
import re
import glob
import os


class Processor:
    def __init__(self, path):
        self.path = path

    def process_dataset(self, csv, cols, labels, new_dir):

        # Remove ID and NaN columns
        keep_cols = cols

        df = csv[keep_cols]

        df = df.rename(columns=labels)

        # remove duplicate entries from dataset
        df.drop_duplicates(inplace=True)

        df['sentiment'].value_counts().plot(
            kind='pie',
            autopct=lambda pct: self.show_autopct(pct, df['sentiment'].value_counts())
        )

        plt.show()
        df.to_csv(new_dir, index=False)

    @staticmethod
    def show_autopct(pct, data):
        absolute = int(pct / 100. * np.sum(data))
        return "{:d}\n({:.1f}%)".format(absolute, pct)

    def process_dataset_taskA(self):
        directory_path = f"{self.path}/A/"

        all_files = glob.glob(os.path.join(directory_path, "trainA*.csv"))

        df_from_each_file = (pd.read_csv(f, sep='\t', header=None) for f in all_files)
        train_df = pd.concat(df_from_each_file, ignore_index=True)

        test_df = pd.read_csv(os.path.join(directory_path, "testA.csv"), sep='\t', header=None)

        keep_cols = [1, 2]

        labels = {1: "sentiment", 2: "message"}

        self.process_dataset(csv=train_df, cols=keep_cols, labels=labels, new_dir="Datasets/task_train_A.csv")

        self.process_dataset(csv=test_df, cols=keep_cols, labels=labels, new_dir="Datasets/task_test_A.csv")

    def process_dataset_taskB(self):
        directory_path = f"{self.path}/B/"

        all_files = glob.glob(os.path.join(directory_path, "trainB*.csv"))

        df_from_each_file = (pd.read_csv(f, sep='\t', header=None) for f in all_files)
        train_df = pd.concat(df_from_each_file, ignore_index=True)

        test_df = pd.read_csv(os.path.join(directory_path, "testB.csv"), sep='\t', header=None)

        keep_cols = [1, 2, 3]

        labels = {1: "topic", 2: "sentiment", 3: "message"}

        self.process_dataset(csv=train_df, cols=keep_cols, labels=labels, new_dir="Datasets/task_train_B.csv")

        self.process_dataset(csv=test_df, cols=keep_cols, labels=labels, new_dir="Datasets/task_test_B.csv")

    @staticmethod
    def clean_message(message):
        stop_words = set(stopwords.words('english'))

        message = message.lower()
        # remove all links and URLs from the tweets
        message = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', message)
        # remove all the usernames if any
        message = re.sub('@[^\s]+', '', message)
        # remove all the hashtag symbols form the tweets
        message = re.sub(r'#([^\s]+)', '', message)
        # Remove punctuations and numbers
        message = re.sub('[^a-zA-Z]', ' ', message)

        # Single character removal
        message = re.sub(r"\s+[a-zA-Z]\s+", ' ', message)

        # Removing multiple spaces
        message = re.sub(r'\s+', ' ', message)

        message = [word for word in word_tokenize(message) if word not in stop_words]
        return message

    def clean_messages(self, messages):
        X = []
        for message in messages:
            message = self.clean_message(message)
            X.append(message)

        return X

    def clean_messages_b(self, messages, topics):
        X = []
        for i in range(len(messages)):
            messages[i] = self.clean_message(messages[i])
            messages[i].append(topics[i])
            X.append(messages[i])

        return X

    @staticmethod
    def feature_extraction(X, y, num_words=5000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        vocab_size = len(tokenizer.word_index) + 1

        X_train = pad_sequences(X_train, maxlen=100, padding='post', truncating='post')

        X_test = pad_sequences(X_test, maxlen=100, padding='post', truncating='post')

        return X_train, X_test, y_train, y_test, vocab_size, tokenizer

    def feature_extraction_a_test(self, data_dir, tokenizer):
        df = pd.read_csv(f"{data_dir}.csv", sep=',')
        X = list(df['message'])
        y = pd.get_dummies(df['sentiment']).values

        X = self.clean_messages(X)

        X_test = tokenizer.texts_to_sequences(X)

        X_test = pad_sequences(X_test, maxlen=100, padding='post', truncating='post')

        return X_test, y

    def feature_extraction_a(self, data_dir):
        df = pd.read_csv(f"{data_dir}.csv", sep=',')
        X = list(df['message'])
        y = pd.get_dummies(df['sentiment']).values

        X = self.clean_messages(X)

        return self.feature_extraction(X, y, num_words=50000)

    def feature_extraction_b_test(self, data_dir, tokenizer):
        df = pd.read_csv(f"{data_dir}.csv", sep=',')
        messages = df['message']
        topics = df['topic']
        y = df['sentiment']
        y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

        X = self.clean_messages_b(messages, topics)

        X_test = tokenizer.texts_to_sequences(X)

        X_test = pad_sequences(X_test, maxlen=100, padding='post', truncating='post')

        return X_test, y

    def feature_extraction_b(self, data_dir):
        df = pd.read_csv(f"{data_dir}.csv", sep=',')
        messages = df['message']
        topics = df['topic']
        y = df['sentiment']
        y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

        X = self.clean_messages_b(messages, topics)

        return self.feature_extraction(X, y)

    @staticmethod
    def embedding_matrix(vocab_size, tokenizer):
        embeddings_dictionary = dict()
        glove_file = open('glove.twitter.27B.200d.txt', encoding="utf8")

        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        embedding_matrix = zeros((vocab_size, 200))
        for word, index in tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix
