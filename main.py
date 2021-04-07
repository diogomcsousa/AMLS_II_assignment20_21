from Utils.data_processing import Processor
from A.taskA import *
from B.taskB import *
import gc

# ======================================================================================================================
# python3 -m nltk.downloader stopwords
# IMPORTANT
# ======================================================================================================================

base_dir = "/Users/diogosousa/Documents/IMLS/Term2/AMLSII/AMLS_II/Assignment_Datasets"

# ======================================================================================================================
# Data preprocessing

processor = Processor(base_dir)
#processor.process_dataset_taskA()  # Pre-process dataset for task A
#processor.process_dataset_taskB()  # Pre-process dataset for task B

# ======================================================================================================================
# Task A
X_train_a, X_test_a, y_train_a, y_test_a, vocab_size_a, tokenizer_a = processor.feature_extraction_a('Datasets/task_train_A')


embedding_matrix_a = processor.embedding_matrix(vocab_size_a, tokenizer_a)

model_A = model_task_a(embedding_matrix_a, vocab_size_a, 200)                 # Build model object.
acc_A_train = fit_a(X_train_a, y_train_a) # Train model based on the training set (you should fine-tune your model based on validation set.)

accuracy_a, f1_a, recall_a, precision_a = predict_a(X_test_a, y_test_a)   # Test model based on the test set.
X_test_extra_a, y_test_extra_a = processor.feature_extraction_a_test('Datasets/task_test_A', tokenizer_a)
accuracy_extra_test_a, f1_extra_test_a, recall_extra_test_a, precision_extra_test_a = predict_a(X_test_extra_a, y_test_extra_a)

print("-----------------------------------------------")
print("Cleaning memory before next task... ")

gc.collect()    # Clean up memory

print("Cleared!")

# ======================================================================================================================
# Task B

X_train_b, X_test_b, y_train_b, y_test_b, vocab_size_b, tokenizer_b = processor.feature_extraction_b('Datasets/task_train_B')

embedding_matrix_b = processor.embedding_matrix(vocab_size_b, tokenizer_b)

model_B = model_task_b(embedding_matrix_b, vocab_size_b, 200)                 # Build model object.
acc_B_train = fit_b(X_train_b, y_train_b) # Train model based on the training set (you should fine-tune your model based on validation set.)

accuracy_b, f1_b, recall_b, precision_b = predict_b(X_test_b, y_test_b)   # Test model based on the test set.
X_test_extra_b, y_test_extra_b = processor.feature_extraction_b_test('Datasets/task_test_B', tokenizer_b)
accuracy_extra_test_b, f1_extra_test_b, recall_extra_test_b, precision_extra_test_b = predict_b(X_test_extra_b, y_test_extra_b)

print("-----------------------------------------------")
print("Cleaning memory before next task... ")

gc.collect()    # Clean up memory

print("Cleared!")

# ======================================================================================================================
## Print out your results with following format:
print('TA:{},{},{};\nTB:{},{},{};'.format(acc_A_train, accuracy_a, accuracy_extra_test_a,
                                          acc_B_train, accuracy_b, accuracy_extra_test_b))


