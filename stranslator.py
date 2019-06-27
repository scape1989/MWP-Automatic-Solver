from __future__ import absolute_import

from keras_transformer import get_model, decode
from keras import callbacks, metrics, optimizers
from keras import backend as K
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import numpy as np
import os
import pickle
import re
import time
import random


DIR_PATH = DIR_PATH = os.path.abspath(os.path.dirname(__file__))
DATASET = 'data/gen1.pickle'
DATA_PATH = os.path.join(DIR_PATH, DATASET)

# Data constraints
VALIDATION_SPLIT = 0.1
TEST_RESERVE = 0.05
EQUALS_SIGN = False

# Hyperparameters
EMBED_DIM = 32
NUM_HEADS = 8
NUM_DECODER = 4
NUM_ENCODER = 4
DFF = 128
DROPOUT = 0.01

EPOCHS = 20
BATCH_SIZE = 16
SHUFFLE = True

# Adam params
BETA_1 = 0.98
BETA_2 = 0.99
EPSILON = 1e-8
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
    # Load binary file and return the array of lines
    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

    return file_data


def get_as_tuple(example):
    # Each example in the data is an array of tuples
    ex_as_dict = dict(example)

    # Return the information we need from the set
    return ex_as_dict["question"], ex_as_dict["equation"]


def log(what):
    # Append to the model's log
    with open(os.path.join(DIR_PATH, f"./logs/{MODEL_NAME}.txt"), 'a+') as fh:
        fh.write(what + '\n')


def expressionize(what):
    # Strip out x =
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


source_tokens = []
target_tokens = []

test_tokens = []
test_target_tokens = []


# Generate dictionaries
examples = read_data_from_file(DATA_PATH)
if len(examples) > 20000:
    examples = examples[:20000]

print(f"{len(examples)} examples loaded.")

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


progress_increment = 1
for example in examples[:train_size]:
    progress = int((progress_increment / len(examples)) * 100)
    progress_increment += 1
    print('Progress: [%d%%]\r' % progress, end="")
    try:
        problem = get_as_tuple(example)
        text = problem[0]
        equation = problem[1]

        source_tokens.append(text)
        # The equation translation
        if not EQUALS_SIGN:
            e = expressionize(equation)
            target_tokens.append(e)
        else:
            target_tokens.append(equation)
    except:
        pass

for example in examples[train_size:]:
    progress = int((progress_increment / len(examples)) * 100)
    progress_increment += 1
    print('Progress: [%d%%]\r' % progress, end="")
    # Encode only the questions for testing the translation
    try:
        problem = get_as_tuple(example)
        text = problem[0]
        equation = problem[1]

        test_tokens.append(text)
        # The equation translation
        if not EQUALS_SIGN:
            e = expressionize(equation)
            test_target_tokens.append(e)
        else:
            test_target_tokens.append(equation)
    except:
        pass
print()

print("Building dataframe...")

progress_increment = 1
# All problems and equations
simple_corpus_sentences = []
combination = source_tokens + test_tokens + target_tokens + test_target_tokens
for problem in combination:
    progress = int((progress_increment / len(combination)) * 100)
    progress_increment += 1
    print('Progress: [%d%%]\r' % progress, end="")
    simple_corpus_sentences.append(problem)
print()

df_clean = pd.DataFrame({'clean': simple_corpus_sentences})
df_clean = df_clean.dropna().drop_duplicates()

# Space split questions and equations
sentences = [row.split() for row in df_clean['clean']]
words = []

print("Extracting words from sentences...")

progress_increment = 1
for sentence in sentences:
    progress = int((progress_increment / len(sentences)) * 100)
    progress_increment += 1
    print('Progress: [%d%%]\r' % progress, end="")
    for word in sentences:
        words.append(word)
print()

print("Training W2V model...")


w2v_model = Word2Vec(min_count=1,
                     window=3,
                     size=1,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=5,
                     seed=SEED)

w2v_model.build_vocab(words)

w2v_model.train(words,
                total_examples=w2v_model.corpus_count,
                epochs=10,
                report_delay=1)

# The question text dictionary
token_dict = {
    "<PAD>": 3,
    "<START>": 1,
    "<END>": 2
}

print("Creating lookup table from vocabulary...")

for sentence in sentences:
    for word in sentence:
        token_dict[word] = w2v_model.wv[word].item(0)


print("Encoding inputs...")


def wrap(sentence_list, start_token=True):
    problems = []

    for sentence in sentence_list:
        arr = []
        if start_token:
            arr.append("<START>")

        for word in sentence.split(' '):
            if len(word) > 0:
                arr.append(word)

        arr.append('<END>')

        if not start_token:
            arr.append("<PAD>")

        problems.append(arr)

    return problems


# Add special tokens
encode_tokens = wrap(source_tokens)
decode_tokens = wrap(target_tokens)
output_tokens = wrap(target_tokens, start_token=False)

# Do the same for the test data
ttokens = wrap(test_tokens)
dttokens = wrap(test_target_tokens)
ottokens = wrap(test_target_tokens, start_token=False)


def serialize_w2v(tokens, dim=False):
    vecProbs = []
    for problem in tokens:
        vecProb = []

        for token in problem:
            embedding = token_dict[token]
            if isinstance(embedding, np.ndarray):
                for vex in np.nditer(embedding):
                    if not dim:
                        vecProb.append(vex.item(0))
                    else:
                        vecProb.append([vex.item(0)])

            else:
                if not dim:
                    vecProb.append(embedding)
                else:
                    vecProb.append([embedding])

        vecProbs.append(vecProb)

    return vecProbs


encoded_input = serialize_w2v(encode_tokens)
decoded_input = serialize_w2v(decode_tokens)
decoded_output = serialize_w2v(output_tokens, dim=True)

tencoded_input = serialize_w2v(ttokens)
tdecoded_input = serialize_w2v(dttokens)
tdecoded_output = serialize_w2v(ottokens, dim=True)

print("Padding the embeddedings...")

# Pad to max length of input set
source_max_len = max(map(len, encoded_input))
target_max_len = max(map(len, decoded_input))
# Padding length for the training set
tsource_max_len = max(map(len, tencoded_input))
ttarget_max_len = max(map(len, tdecoded_input))

# Padding for each encoding
encoded_input = [tokens + [token_dict["<PAD>"]] *
                 (source_max_len - len(tokens)) for tokens in encoded_input]
decoded_input = [tokens + [token_dict["<PAD>"]] *
                 (target_max_len - len(tokens)) for tokens in decoded_input]
output_tokens = [tokens + [[token_dict["<PAD>"]]] *
                 (target_max_len - len(tokens)) for tokens in decoded_output]


tencoded_input = [tokens + [token_dict["<PAD>"]] *
                  (tsource_max_len - len(tokens)) for tokens in tencoded_input]
tdecoded_input = [tokens + [token_dict["<PAD>"]] *
                  (ttarget_max_len - len(tokens)) for tokens in tdecoded_input]
toutput_tokens = [tokens + [[token_dict["<PAD>"]]] *
                  (ttarget_max_len - len(tokens)) for tokens in tdecoded_output]

print("...done.")

print("Building the Transformer...")
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

# Build & fit model
model = get_model(
    token_num=len(token_dict),
    embed_dim=EMBED_DIM,
    encoder_num=NUM_ENCODER,
    decoder_num=NUM_DECODER,
    head_num=NUM_HEADS,
    hidden_dim=DFF,
    dropout_rate=DROPOUT,
    use_same_embed=True
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
X = [np.array(encoded_input), np.array(decoded_input)]
y = np.array(output_tokens)

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

print("Evaluating on test data...")

Xt = [np.array(tencoded_input), np.array(tdecoded_input)]
yt = np.array(toutput_tokens)

model.evaluate(x=Xt,
               y=yt)

# Predict with Beam Search
decoded = decode(
    model,
    tencoded_input,
    start_token=token_dict['<START>'],
    end_token=token_dict['<END>'],
    pad_token=token_dict['<PAD>'],
    top_k=10,
    temperature=1.0,
)

# print(decoded)
# Make predictions
# for i in range(len(test_tokens) - 1):
#    prediction = ' '.join(decoded[i][1:-1])
#    print(prediction)
#    log(prediction)
