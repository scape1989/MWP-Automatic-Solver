from __future__ import absolute_import

from keras_transformer import get_model, decode
from keras import callbacks, metrics, optimizers
from keras import backend as K
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import os
import pickle
import re
import time
import random


DIR_PATH = DIR_PATH = os.path.abspath(os.path.dirname(__file__))
DATASET = 'data/prefix_data.pickle'
DATA_PATH = os.path.join(DIR_PATH, DATASET)

# Data constraints
TEST_SPLIT = 0.05
VALIDATION_SPLIT = 0.1
TEST_RESERVE = 0.05
EQUALS_SIGN = False
BUFFER_SIZE = 400000
MAX_LENGTH = 40

# Hyperparameters
EMBED_DIM = 32
NUM_HEADS = 8
NUM_DECODER = 2
NUM_ENCODER = 2
DFF = 256
DROPOUT = 0.05

EPOCHS = 20
BATCH_SIZE = 32
SHUFFLE = True

# Adam params
BETA_1 = 0.98
BETA_2 = 0.99
EPSILON = 1e-9
LEARNING = 0.001

# Random seed for consistency
SEED = 1234

# The name to keep track of any changes
MODEL_NAME = f"t{EPOCHS}_{DFF}_{NUM_HEADS}_{BATCH_SIZE}_{int(time.time())}"
TRAINED_PATH = os.path.join(DIR_PATH,
                            f'models/trained/{MODEL_NAME}/')
LOG_FILE = os.path.join(DIR_PATH,
                        f'logs/{MODEL_NAME}.csv')
MODEL_FILE = os.path.join(DIR_PATH,
                          f'models/trained/{MODEL_NAME}.h5')

random.seed(SEED)


def read_data_from_file(path):
    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

    return file_data


def get_as_tuple(example):
    ex_as_dict = dict(example)

    return ex_as_dict["question"], ex_as_dict["equation"]


def log(what):
    # Append to the model's log
    with open(os.path.join(DIR_PATH, f"./logs/{MODEL_NAME}.txt"), 'a+') as fh:
        fh.write(what + '\n')


def expressionize(what):
    return re.sub(r"([a-z] \=|\= [a-z])", "", what)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


print('Starting the Math Word Problem (MWP) Transformer training.')
print(f'\nTokenizing data from {DATASET}...\n')

examples = read_data_from_file(DATA_PATH)

train_text = []
train_equations = []

val_text = []
val_equations = []

split_point = int((1 - TEST_SPLIT) * len(examples))

for example in examples[:split_point]:
    try:
        # The text
        train_text.append(get_as_tuple(example)[0])
        # The equation translation
        if not EQUALS_SIGN:
            e = expressionize(get_as_tuple(example)[1])

            train_equations.append(e)
        else:
            train_equations.append(get_as_tuple(example)[1])

    except:
        # Some of the filtered questions are empty... Working on a fix
        pass

for example in examples[split_point + 1:]:
    try:
            # The text
        val_text.append(get_as_tuple(example)[0])
        # The equation translation
        if not EQUALS_SIGN:
            e = expressionize(get_as_tuple(example)[1])

            val_equations.append(e)
        else:
            val_equations.append(get_as_tuple(example)[1])
    except:
        pass

print(f"Data split <- {len(train_text)}:{len(val_text)}")

# Convert arrays to TensorFlow constants
train_text_const = tf.constant(train_text)
train_eq_const = tf.constant(train_equations)
test_text_const = tf.constant(val_text)
test_eq_const = tf.constant(val_equations)

# Turn the constants into TensorFlow Datasets
# Training
t_dataset = tf.data.Dataset.from_tensor_slices((train_text_const,
                                                train_eq_const))
# Testing
test_dataset = tf.data.Dataset.from_tensor_slices((test_text_const,
                                                   test_eq_const))

tokenizer_txt = tfds.features.text.ByteTextEncoder.build_from_corpus(
    (txt.numpy() for txt, eq in t_dataset), target_vocab_size=2**13)

tokenizer_eq = tfds.features.text.ByteTextEncoder.build_from_corpus(
    (eq.numpy() for txt, eq in t_dataset), target_vocab_size=2**13)


def encode(lang1, lang2):
    lang1 = [tokenizer_txt.vocab_size] + tokenizer_txt.encode(
        lang1.numpy()) + [tokenizer_txt.vocab_size + 1]

    lang2 = [tokenizer_eq.vocab_size] + tokenizer_eq.encode(
        lang2.numpy()) + [tokenizer_eq.vocab_size + 1]

    return lang1, lang2


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def tf_encode(txt, eq):
    return tf.py_function(encode, [txt, eq], [tf.int64, tf.int64])


train_dataset = t_dataset.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# Cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,
                                                                padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.map(tf_encode)
test_dataset = test_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE,
                                                                   padded_shapes=([-1], [-1]))

input_vocab_size = tokenizer_txt.vocab_size + 2
target_vocab_size = tokenizer_eq.vocab_size + 2

print('...done.')

print('\nDefining the Transformer model...')
log(DATASET)
log(f"Data split <- {len(train_text)}:{len(val_text)}")
log(f"\nEpochs: {EPOCHS}")
log(f"Equals Sign: {EQUALS_SIGN}")
log(f"Heads: {NUM_HEADS}")
log(f"Encoders: {NUM_ENCODER}")
log(f"Decoders: {NUM_DECODER}")
log(f"Batch Size: {BATCH_SIZE}")
log(f"Feed Forward Depth: {DFF}")
log(f"Dropout: {DROPOUT}")
log(f"Shuffle: {SHUFFLE}")


# Build & fit model
model = get_model(
    token_num=max(input_vocab_size, target_vocab_size),
    embed_dim=EMBED_DIM,
    encoder_num=NUM_ENCODER,
    decoder_num=NUM_DECODER,
    head_num=NUM_HEADS,
    hidden_dim=DFF,
    dropout_rate=DROPOUT,
    use_same_embed=False
)

optimizer = optimizers.Adam(lr=LEARNING,
                            beta_1=BETA_1,
                            beta_2=BETA_2,
                            epsilon=EPSILON)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[f1])

print("...Done.")

model.summary()

print("Training...")

model.fit(train_dataset,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          shuffle=SHUFFLE,
          validation_split=VALIDATION_SPLIT,
          callbacks=[callbacks.ModelCheckpoint(TRAINED_PATH,
                                               monitor='val_acc',
                                               save_best_only=True),
                     callbacks.TensorBoard(log_dir='./tensorboard',
                                           batch_size=BATCH_SIZE),
                     callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2,
                                                 patience=5,
                                                 min_lr=0.001),
                     callbacks.CSVLogger(LOG_FILE,
                                         append=True), ]
          )

print("...Done.")

model.save(MODEL_FILE)

print("Model saved.")


# Predict with Beam Search
decoded = decode(
    model,
    test_dataset,
    start_token=tokenizer_txt.vocab_size,
    end_token=tokenizer_txt.vocab_size + 1,
    pad_token="<pad>",
    top_k=10,
    temperature=1.0,
)

print(decoded)
