# === Imports and Setup ===
import nltk
import json
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset


#! change the following constants as needed
EPOCHS = 200
USE_EXTERNAL_DATASET = True  # set to False to use intents.json


# download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!']

# load the dataset...

# check to see if the local dataset should be used or the ULM dataset
if USE_EXTERNAL_DATASET:
    print("[ ! ] Using external dataset.")
    #! use the defined dataset, intents are specific to the dataset
    ds = load_dataset("OpenAssistant/oasst1")
    intents = {
        "intents": [
            {
                "tag": "chat",
                "patterns": [item['text'] for item in ds['train'] if item['role'] == 'prompter'],
                "responses": [item['text'] for item in ds['train'] if item['role'] == 'assistant']
            }
        ]
    }
else:
    print("[ ! ] Using intents.json dataset.")
    with open('intents.json') as file:
        intents = json.load(file)

# look for unique words and classes
words, classes, documents = [], [], []

for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, tag))
        if tag not in classes:
            classes.append(tag)

words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words])))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
pickle.dump(documents, open('documents.pkl', 'wb'))
pickle.dump(intents, open('intents.pkl', 'wb'))

# prepare training data
training_data = []
output_template = [0] * len(classes)

for tokens, tag in documents:
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = output_template.copy()
    output_row[classes.index(tag)] = 1
    training_data.append([bag, output_row])

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
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='loss', patience=EPOCHS, verbose=1)
model_checkpoint = ModelCheckpoint('bestchatbot_model.h5', monitor='loss', save_best_only=True, verbose=1)

# fit the model
history = model.fit(np.array(train_x), np.array(train_y),
                    epochs=300, batch_size=5, verbose=1,
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