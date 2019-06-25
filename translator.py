from __future__ import absolute_import

from keras_transformer import get_model, decode
from keras import callbacks, metrics, optimizers
from keras import backend as K
from bert_embedding import BertEmbedding
import numpy as np
import os
import pickle
import re
import time
import random

bert_embedding = BertEmbedding(model='bert_24_1024_16',
                               dataset_name='book_corpus_wiki_en_uncased')


DIR_PATH = DIR_PATH = os.path.abspath(os.path.dirname(__file__))
DATASET = 'data/prefix_data.pickle'
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
BATCH_SIZE = 16
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

# Generate dictionaries
examples = read_data_from_file(DATA_PATH)

# Mix up the problems
print(f"Shuffling data with seed: {SEED}")
random.shuffle(examples)

train_size = len(examples) - int(TEST_RESERVE * len(examples))
log(f"{len(examples) - train_size}/{train_size + (len(examples) - train_size)} problems reverved for testing.")

print("Tokenizing data...")


def reduced_representation(embedding):
    # Due to memory constraints, use only the first element in each
    #  representation array given by BERT
    # ! Using the full array does work if you have the capacity
    small_em = []

    for token in embedding:
        try:
            tok = "".join(token[0])
            rep = token[1][0].item(0)
            pair = (tok, rep)
            small_em.append(pair)
        except:
            pass
    return small_em


def build_token_dict(problem_list):
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
    }

    for problem in problem_list:
        for tokens in problem:
            word = tokens[0]
            embedding = tokens[1]

            if word not in token_dict:
                # Assign a lookup to a piece of the BERT representation
                token_dict[word] = embedding

    return token_dict


problem_number = 1
for example in examples[:train_size]:
    progress = int(problem_number / len(examples) * 100)
    print('Progress: [%d%%]\r' % progress, end="")
    try:
        problem = get_as_tuple(example)
        text = problem[0]
        equation = problem[1]
        # The text
        mwp = text.split(' ')
        mwp = reduced_representation(bert_embedding(mwp))
        source_tokens.append(mwp)
        # The equation translation
        if not EQUALS_SIGN:
            e = expressionize(equation).split(' ')
            e = reduced_representation(bert_embedding(e))
            target_tokens.append(e)
        else:
            e = equation.split(' ')
            e = reduced_representation(bert_embedding(e))
            target_tokens.append(e)
    except:
        pass
    problem_number += 1

for example in examples[train_size:]:
    progress = int(problem_number / len(examples) * 100)
    print('Progress: [%d%%]\r' % progress, end="")
    # Encode only the questions for testing the translation
    try:
        text = get_as_tuple(example)[0]
        # The text
        mwp = text.split(' ')
        test_tokens.append(mwp)
    except:
        pass
    problem_number += 1

print("...done.")

print("Building lookup table from vocabulary...")

# Token to np arrays
# The question text dictionary
source_token_dict = build_token_dict(source_tokens)
# The equation dictionary
target_token_dict = build_token_dict(target_tokens)

print("...done.")


def get_bert_encoded_input(problems, output_embedding=False):
    encoded_tokens = []

    for problem in problems:
        if output_embedding == False:
            p = [source_token_dict["<START>"]]
        else:
            p = []

        for token in problem:
            # (token, value)
            # Append only the representation of the word
            if output_embedding == False:
                p.append(token[1])
            else:
                p.append([token[1]])

        if output_embedding:
            p.append([source_token_dict["<END>"]])
            p.append([source_token_dict["<PAD>"]])
        else:
            p.append(source_token_dict["<END>"])

        encoded_tokens.append(p)

    return encoded_tokens


print("Building reverse lookup table...")

# Problem vocab lookup table
target_token_dict_inv = {}
for k, v in target_token_dict.items():
    # Value is array of np arrays
    target_token_dict_inv[f'{v}'] = k

encoded_input = get_bert_encoded_input(source_tokens)
decoded_input = get_bert_encoded_input(target_tokens)

output_embedding = get_bert_encoded_input(target_tokens, output_embedding=True)

print("...done.")

# Pad to max length of input sets
source_max_len = max(map(len, encoded_input))
target_max_len = max(map(len, decoded_input))

print("Padding the embedded problem text...")

# Padding for each encoding
encoded_input = [tokens + [source_token_dict["<PAD>"]] *
                 (source_max_len - len(tokens)) for tokens in encoded_input]
decoded_input = [tokens + [source_token_dict["<PAD>"]] *
                 (target_max_len - len(tokens)) for tokens in decoded_input]

print("Padding the embedded expressions...")
output_embedding = [tokens + [[source_token_dict["<PAD>"]]] *
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

optimizer = optimizers.Adam(lr=LEARNING,
                            beta_1=BETA_1,
                            beta_2=BETA_2,
                            epsilon=EPSILON)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[f1])

print("...Done.")

model.summary()

# Input shapes
X = [np.array(encoded_input * 1024), np.array(decoded_input * 1024)]
y = np.array(output_embedding * 1024)

print(X[0].shape, X[1].shape, y.shape)

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
                                         append=True), ]
          )

print("...Done.")

model.save(MODEL_FILE)

print("Model saved.")

print(test_input)

# Predict with Beam Search
decoded = decode(
    model,
    test_input,
    start_token=target_token_dict['<START>'],
    end_token=target_token_dict['<END>'],
    pad_token=target_token_dict['<PAD>'],
    top_k=10,
    temperature=1.0,
)

print(decoded)
exit()

# Make predictions
for i in range(len(test_tokens) - 1):
    prediction = ''.join(
        map(lambda x: target_token_dict_inv[x], decoded[i][1:-1]))
    print(prediction)
    log(prediction)
