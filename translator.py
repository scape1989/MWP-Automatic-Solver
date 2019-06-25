from __future__ import absolute_import

from keras_transformer import get_model, decode
from keras import callbacks, metrics, optimizers
from keras import backend as K
from bert_embedding import BertEmbedding
from models.transformer.CustomSchedule import CustomSchedule
import numpy as np
import os
import pickle
import re
import time
import random

bert_embedding = BertEmbedding(model='bert_24_1024_16',
                               dataset_name='book_corpus_wiki_en_uncased')


DIR_PATH = DIR_PATH = os.path.abspath(os.path.dirname(__file__))
DATASET = 'data/small_data.pickle'
DATA_PATH = os.path.join(DIR_PATH, DATASET)

# Data constraints
VALIDATION_SPLIT = 0.1
TEST_RESERVE = 0.05
EQUALS_SIGN = False

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
BETA_1 = 0.9
BETA_2 = 0.98
EPSILON = 1e-9

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


def build_token_dict(token_list):
    token_dict = {
        '<PAD>': [np.array([0])],
        '<START>': [np.array([1])],
        '<END>': [np.array([2])],
    }

    for tokens in token_list:
        for token in tokens:
            word = ''.join(token[0])
            embeddings = token[1]

            if word not in token_dict:
                # Assign a lookup to an np array representation
                token_dict[word] = embeddings

    return token_dict


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


class F1():
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


log(DATASET)
log(f"\nEpochs: {EPOCHS}")
log(f"Equals Sign: {EQUALS_SIGN}")
log(f"Heads: {NUM_HEADS}")
log(f"Encoders: {NUM_ENCODER}")
log(f"Decoders: {NUM_DECODER}")
log(f"Batch Size: {BATCH_SIZE}")
log(f"Feed Forward Depth: {DFF}")
log(f"Dropout: {DROPOUT}")
log(f"Shuffle: {SHUFFLE}")

source_tokens = []
target_tokens = []

test_tokens = []
ttest_tokens = []

# Generate dictionaries
examples = read_data_from_file(DATA_PATH)

# Mix up the problems
print(f"Shuffling data with seed: {SEED}")
random.shuffle(examples)

train_size = len(examples) - int(TEST_RESERVE * len(examples))
log(f"{len(examples) - train_size}/{train_size + (len(examples) - train_size)} problems reverved for testing.")

print("Tokenizing data...")

for example in examples[:1]:
    try:
        problem = get_as_tuple(example)
        text = problem[0]
        equation = problem[1]
        # The text
        mwp = text.split(' ')
        mwp = bert_embedding(mwp)
        source_tokens.append(mwp)
        # The equation translation
        if not EQUALS_SIGN:
            e = expressionize(equation).split(' ')
            e = bert_embedding(e)
            target_tokens.append(e)
        else:
            e = equation.split(' ')
            e = bert_embedding(e)
            target_tokens.append(e)
    except:
        pass

# Token to np arrays
source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)


print("...done.")


def get_bert_encoded_input(tokens, output_embedding=False):
    encoded_tokens = []

    for problem_words in tokens:
        if not output_embedding:
            problem = [source_token_dict["<START>"][0]]
        else:
            problem = []

        for tokens in problem_words:
            representations = tokens[1]
            if len(representations) > 0:
                for nparray in representations:
                    problem.append(nparray)

        problem.append(source_token_dict["<END>"][0])
        if output_embedding:
            problem.append(source_token_dict["<PAD>"][0])

        encoded_tokens.append(problem)

    return encoded_tokens


print("Building BERT vocab lookup table...")

# Problem vocab lookup table
target_token_dict_inv = {}
for k, v in target_token_dict.items():
    # Value is array of np arrays
    for array in v:
        key = str(array)
        key = re.sub('\n', '', key)
        target_token_dict_inv[f'{key}'] = k


encoded_input = get_bert_encoded_input(source_tokens)
decoded_input = get_bert_encoded_input(target_tokens)

output_embedding = get_bert_encoded_input(target_tokens, output_embedding=True)

print("...done.")

# Pad to max length of input sets
source_max_len = max(map(len, encoded_input))
target_max_len = max(map(len, decoded_input))

print("Padding the embedded problem text...")

# Padding for each encoding
encoded_input = [tokens + [source_token_dict["<PAD>"][0]] *
                 (source_max_len - len(tokens)) for tokens in encoded_input]
decoded_input = [tokens + [source_token_dict["<PAD>"][0]] *
                 (target_max_len - len(tokens)) for tokens in decoded_input]

print("Padding the embedded expressions...")
output_embedding = [tokens + [source_token_dict["<PAD>"][0]] *
                    (target_max_len - len(tokens)) for tokens in output_embedding]

print("...Done.")

print("Building the Transformer...")
# Build & fit model
model = get_model(
    token_num=max(len(source_token_dict), len(target_token_dict)),
    embed_dim=EMBED_DIM,
    encoder_num=NUM_ENCODER,
    decoder_num=NUM_DECODER,
    head_num=NUM_HEADS,
    hidden_dim=DFF,
    dropout_rate=DROPOUT,
    use_same_embed=True,  # Use different embeddings for different languages
)

optimizer = optimizers.Adam(lr=0.001,
                            beta_1=BETA_1,
                            beta_2=BETA_2,
                            epsilon=EPSILON,
                            decay=0.9999)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[f1]
              )

print("...Done.")

model.summary()

# Input shapes
X = [np.array(encoded_input * 1024), np.array(decoded_input * 1024)]
y = [np.array(output_embedding * 1024), None]

print("Training...")

model.fit(x=X,
          y=y,
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
                                         append=True),
                     F1((X, y))]
          )

print("...Done.")

model.save(MODEL_FILE)

print("Model saved.")

# Predict with Beam Search
decoded = decode(
    model,
    encoded_input,
    start_token=target_token_dict['<START>'],
    end_token=target_token_dict['<END>'],
    pad_token=target_token_dict['<PAD>'],
    top_k=10,
    temperature=1.0,
)

# Make predictions
for i in range(10):
    prediction = ''.join(
        map(lambda x: target_token_dict_inv[x], decoded[i][1:-1]))
    print(prediction)
    log(prediction)
