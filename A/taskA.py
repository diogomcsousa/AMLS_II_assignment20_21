from tensorflow.keras import layers, models
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np


def model_task_a(word_embeddings, vocab_size, embedding_size):
    model = models.Sequential()
    model.add(
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[word_embeddings], input_length=100,
                         trainable=False))

    model.add(layers.BatchNormalization())
    model.add(layers.GaussianNoise(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Bidirectional(layers.LSTM(units=150, return_sequences=True, recurrent_dropout=0.3)))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(units=150, return_sequences=True, recurrent_dropout=0.3)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('Models/A/dual_lstm')
    return model


def fit_a(X_train, y_train):
    es = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    model = models.load_model('Models/A/dual_lstm')

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, verbose=1, batch_size=128, callbacks=[es])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    model.save('Models/A/dual_lstm')
    return history.history['accuracy'][-1]


def predict_a(X_test, y_test):
    model = models.load_model('Models/A/dual_lstm')
    y_test_final = pd.DataFrame(data=y_test).idxmax(axis=1).values
    y_pred = np.argmax(model.predict(X_test), axis=-1)


    accuracy = accuracy_score(y_test_final, y_pred)
    f1 = f1_score(y_test_final, y_pred, average='macro')
    recall = recall_score(y_test_final, y_pred, average='macro')
    precision = precision_score(y_test_final, y_pred, average='macro')

    cm = confusion_matrix(y_test_final, y_pred, normalize='true')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.get_cmap('Blues', 6))
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['negative', 'neutral', 'positive'], rotation=45)
    ax.set_yticklabels([''] + ['negative', 'neutral', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("score_test: %.4f" % accuracy)
    print("f1_test: %.4f" % f1)
    print("recall_test: %.4f" % recall)
    print("precision_test: %.4f" % precision)

    return accuracy, f1, recall, precision
