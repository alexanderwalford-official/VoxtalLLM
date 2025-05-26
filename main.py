import os
import nltk
import json
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import datasets as importer
from joblib import Parallel, delayed


def process_pattern(pattern, tag):
    tokens = nltk.word_tokenize(pattern)
    return tokens, tag

def process_document(document, words, classes, output_template, lemmatizer):
    tokens, tag = document
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = output_template.copy()
    output_row[classes.index(tag)] = 1
    return [bag, output_row]

#! change the following constants as needed
EPOCHS = 20
USE_EXTERNAL_DATASET = True  # set to False to use intents.json
dataset_url = "OpenAssistant/oasst1"

# download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!']

# load the dataset...

# check to see if the local dataset should be used or the ULM dataset
if os.path.exists('intents.pkl'):
    print("[ ! ] intents already exists, using it.")
    intents = pickle.load(open('intents.pkl', 'rb'))
else:
    if USE_EXTERNAL_DATASET:
        print("[ ! ] Using external dataset.")
        #! use the defined dataset, intents are specific to the dataset
        ds = importer.load_dataset(dataset_url)
        print(str(ds.num_rows) + " rows in the dataset.")
        print("[ ! ] Converting dataset to intents format...")
        intents = {
            "intents": [
                {
                    "tag": "chat",
                    "patterns": [item['text'] for item in tqdm(ds['train']) if item['role'] == 'prompter'],
                    "responses": [item['text'] for item in tqdm(ds['train']) if item['role'] == 'assistant']
                }
            ]
        }
        print("[ ! ] Dataset loaded successfully.")
    else:
        print("[ ! ] Using intents.json dataset.")
        with open('intents.json') as file:
            intents = json.load(file)

    # save the intents to a pickle file for later use
    pickle.dump(intents, open('intents.pkl', 'wb'))

# look for unique words and classes
words, classes, documents = [], [], []

print("[ ! ] Processing intents...")
# iterate through each intent and extract words and classes
for intent in intents['intents']:
    tag = intent['tag']
    if tag not in classes:
        classes.append(tag)
    
    patterns = intent['patterns']
    
    # Parallel execution per intent
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_pattern)(pattern, tag) for pattern in tqdm(patterns)
    )
    
    for tokens, tag in results:
        words.extend(tokens)
        documents.append((tokens, tag))

pickle.dump(documents, open('documents.pkl', 'wb'))

if os.path.exists('words.pkl'):
    print("[ ! ] words already exist, using them.")
    words = pickle.load(open('words.pkl', 'rb'))
else:
    words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words])))
    pickle.dump(words, open('words.pkl', 'wb'))
if os.path.exists('classes.pkl'):
    print("[ ! ] classes already exist, using them.")
    classes = pickle.load(open('classes.pkl', 'rb'))
else:
    classes = sorted(list(set(classes)))
    pickle.dump(classes, open('classes.pkl', 'wb'))

# prepare training data
training_data = []
output_template = [0] * len(classes)

# use all cores
results = Parallel(n_jobs=-1)(
    delayed(process_document)(doc, words, classes, output_template, lemmatizer) for doc in tqdm(documents)
)

training_data = results  # results is a list of [bag, output_row]

# shuffle the training data
random.shuffle(training_data)
training_data = np.array(training_data, dtype=object) # convert to numpy array with object type to handle variable-length lists

# prepare training data for the model
train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])

print("[ ! ] Training data prepared.")

# create the model
model = Sequential([
    Dense(1008, input_shape=(len(train_x[0]),), activation='relu'), # layer 1: input layer with 1008 neurons
    Dropout(0.5), # dropout layer to prevent overfitting
    Dense(512, activation='relu'), # layer 2: hidden layer with 512 neurons
    Dropout(0.5), # dropout layer to prevent overfitting
    Dense(256, activation='relu'), # layer 3: hidden layer with 256 neurons
    Dropout(0.5), # dropout layer to prevent overfitting
    Dense(len(train_y[0]), activation='softmax') # rec activation function is softmax for multi-class classification
])

# define the optimiser and compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='loss', patience=EPOCHS, verbose=1)
model_checkpoint = ModelCheckpoint('bestchatbot_model.h5', monitor='loss', save_best_only=True, verbose=1)

# fit the model
history = model.fit(np.array(train_x), np.array(train_y),
                    epochs=EPOCHS, batch_size=5, verbose=1,
                    callbacks=[early_stopping, model_checkpoint])

# save model
model.save('chatbot_model.h5')
print("[ ! ] Model saved.")


# plot training history


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')

plt.tight_layout()
plt.show()