from tensorflow.keras import layers, models
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


def model_task_b(word_embeddings, vocab_size, embedding_size):
    model = models.Sequential()
    model.add(
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[word_embeddings], input_length=100, trainable=False))

    model.add(layers.BatchNormalization())
    model.add(layers.GaussianNoise(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Bidirectional(layers.LSTM(units=150, return_sequences=True, recurrent_dropout=0.3)))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(units=150, return_sequences=True, recurrent_dropout=0.3)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activity_regularizer=l2(0.0001)))
    model.add(layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.save('Models/B/dual_lstm')
    return model


def fit_b(X_train, y_train):
    es = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    model = models.load_model('Models/B/dual_lstm')

    data_majority = y_train[y_train[:] == 1]
    data_minority = y_train[y_train[:] == 0]

    # Defining the bias
    bias = data_minority.shape[0] / data_majority.shape[0]

    class_weights = {0: 1/bias,
                     1: 1}

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, verbose=1, batch_size=128, class_weight=class_weights, callbacks=[es])

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
    model.save('Models/B/dual_lstm')
    return history.history['accuracy'][-1]


def predict_b(X_test, y_test):
    model = models.load_model('Models/B/dual_lstm')

    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.get_cmap('Blues', 6))
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['positive', 'negative'], rotation=45)
    ax.set_yticklabels([''] + ['positive', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("score_test: %.4f" % accuracy)
    print("f1_test: %.4f" % f1)
    print("recall_test: %.4f" % recall)
    print("precision_test: %.4f" % precision)

    return accuracy, f1, recall, precision
