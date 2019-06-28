from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
# The model
from models.transformer.MultiHeadAttention import MultiHeadAttention
from models.transformer.EncoderLayer import EncoderLayer
from models.transformer.Encoder import Encoder
from models.transformer.DecoderLayer import DecoderLayer
from models.transformer.Decoder import Decoder
from models.transformer.Transformer import Transformer
from models.transformer.CustomSchedule import CustomSchedule
from models.transformer.network import create_masks, loss_function
from keras import backend as K
from nltk.translate.bleu_score import sentence_bleu
# Utilities
import time
import numpy as np
import os
import re
import pickle
import random


DIR_PATH = os.path.abspath(os.path.dirname(__file__))
# Select the dataset to train with
DATASET = 'data/gen1.pickle'
DATA_PATH = os.path.join(DIR_PATH, DATASET)
VALIDATION_SPLIT = 0.1
TEST_RESERVE = 0.05
EQUALS_SIGN = False

# SubwordTextEncoder / TextEncoder
ENCODE_METHOD = tfds.features.text.SubwordTextEncoder

# Data constraints
BUFFER_SIZE = 20000
MAX_LENGTH = 50

# Hyperparameters
NUM_LAYERS = 6
D_MODEL = 64
DFF = 1024
NUM_HEADS = 8
DROPOUT = 0.05

# Training settings
EPOCHS = 20
BATCH_SIZE = 32

# Adam optimizer params
BETA_1 = 0.9
BETA_2 = 0.98
EPSILON = 1e-9

# Random seed for shuffling the data
SEED = 1234

# The name to keep track of any changes
MODEL_NAME = f"x{EPOCHS}_{DFF}_{NUM_HEADS}_{BATCH_SIZE}_{int(time.time())}"
# The checkpoint file where the trained weights will be saved
# Only saves on finish
MODEL_FILE = os.path.join(DIR_PATH,
                          f'models/trained/{MODEL_NAME}.ckpt')

# Set the seed for random
random.seed(SEED)


def read_data_from_file(path):
    # Get the lines in the binary
    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

    return file_data


def get_as_tuple(example):
    # Separate the trainable data
    ex_as_dict = dict(example)

    return ex_as_dict["question"], ex_as_dict["equation"]


def log(what):
    # Append to the model's log
    with open(os.path.join(DIR_PATH, f"./logs/{MODEL_NAME}.txt"), 'a+') as fh:
        fh.write(what + '\n')


def expressionize(what):
    # It may help training if the 'x =' is not learned
    return re.sub(r"([a-z] \=|\= [a-z])", "", what)


def plog(what):
    # Print then log
    print(what)
    log(what)


def main():
    print("Starting the Math Word Problem (MWP) Transformer training.")
    print(f"\nTokenizing data from {DATASET}...\n")

    examples = read_data_from_file(DATA_PATH)

    print(f"Shuffling data with seed: {SEED}")
    random.shuffle(examples)

    train_text = []
    train_equations = []

    val_text = []
    val_equations = []

    split_point = int((1 - VALIDATION_SPLIT) * len(examples))

    # Get training examples
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

    # Sample section of the training data for testing
    # Remove the samples from the training data
    test_split = int(TEST_RESERVE * len(train_text))
    if test_split > 50:
        test_split = 50

    test_text = train_text[:test_split]
    train_text = train_text[test_split:]
    test_equations = train_equations[:test_split]
    train_equations = train_equations[test_split:]

    # Get validation examples
    for example in examples[split_point:]:
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

    print(f"Data split <- {len(train_text)}:{len(val_text)}:{len(test_text)}")

    # Convert arrays to TensorFlow constants
    train_text_const = tf.constant(train_text)
    train_eq_const = tf.constant(train_equations)
    val_text_const = tf.constant(val_text)
    val_eq_const = tf.constant(val_equations)

    # Turn the constants into TensorFlow Datasets
    # Training
    t_dataset = tf.data.Dataset.from_tensor_slices((train_text_const,
                                                    train_eq_const))
    # Validation
    v_dataset = tf.data.Dataset.from_tensor_slices((val_text_const,
                                                    val_eq_const))

    print("Building vocabulary...")

    # Create data tokenizers
    tokenizer_txt = ENCODE_METHOD.build_from_corpus(
        (txt.numpy() for txt, eq in t_dataset), target_vocab_size=2**15)

    tokenizer_eq = ENCODE_METHOD.build_from_corpus(
        (eq.numpy() for txt, eq in t_dataset), target_vocab_size=2**15)

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

    print("Encoding inputs...")

    train_dataset = t_dataset.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # Cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.padded_batch(BATCH_SIZE,
                                               padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = v_dataset.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE,
                                                                     padded_shapes=([-1], [-1]))

    input_vocab_size = tokenizer_txt.vocab_size + 2
    target_vocab_size = tokenizer_eq.vocab_size + 2

    print("...done.")
    print("\nDefining the Transformer model...")

    # Using the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(D_MODEL),
                                         beta_1=BETA_1,
                                         beta_2=BETA_2,
                                         epsilon=EPSILON)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_acc')

    def print_epoch(epoch, batch):
        epoch_information = "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
            epoch + 1, batch, train_loss.result(), train_acc.result())
        # Overwrite the line to see live updated results
        print(f"{epoch_information}\r", end="")

    transformer = Transformer(NUM_LAYERS,
                              D_MODEL,
                              NUM_HEADS,
                              DFF,
                              input_vocab_size,
                              target_vocab_size,
                              DROPOUT)

    print('...done.')
    print('\nTraining...')

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                         tar_inp)

        with tf.GradientTape() as tape:
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)

            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss,
                                  transformer.trainable_variables)

        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))

        train_loss(loss)
        train_acc(tar_real, predictions)

    # Log all the settings used in the session
    log(MODEL_NAME)
    log(MODEL_FILE)
    log(DATASET)
    log(f"Data split <- {len(train_text)}:{len(val_text)}:{len(test_text)}")
    log(f"Random Shuffle Seed: {SEED}")
    log(f"\nEpochs: {EPOCHS}")
    log(f"Batch Size: {BATCH_SIZE}")
    log(f"Buffer Size: {BUFFER_SIZE}")
    log(f"Max Length: {MAX_LENGTH}")
    log(f"Equals Sign: {EQUALS_SIGN}")
    log(f"Layers: {NUM_LAYERS}")
    log(f"Heads: {NUM_HEADS}")
    log(f"Model Depth: {D_MODEL}")
    log(f"Feed Forward Depth: {DFF}")
    log(f"Dropout: {DROPOUT}\n")
    log(f"Adam Params: b1{BETA_1} b2{BETA_2} e{EPSILON}\n")

    # Train
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_acc.reset_states()

        # inp -> MWP, tar -> Equation
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            print_epoch(epoch, batch)

        if epoch == (EPOCHS - 1):
            # Save a checkpoint of model weights
            transformer.save_weights(MODEL_FILE)
            print(f'Saved {MODEL_NAME}\n')

        print_epoch(epoch, batch)
        # Save a log of the epoch results
        log(f"Epoch {epoch + 1}: loss {train_loss.result()} acc {train_acc.result()}")

        # Clear the line being overwritten by print_epoch
        print()
        # Calculate the time the epoch took to complete
        # The first epoch seems to take significantly longer than the others
        print(f'Epoch took {int(time.time() - start)}s\n')

    print('...done.')
    print(f'\nTesting translations after {EPOCHS} epochs...')

    def evaluate(inp_sentence):
        start_token = [tokenizer_txt.vocab_size]
        end_token = [tokenizer_txt.vocab_size + 1]

        # The input is a MWP, hence adding the start and end token
        inp_sentence = start_token + \
            tokenizer_txt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # The target is an equation, the first word to the transformer should be the
        # equation start token.
        decoder_input = [tokenizer_eq.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input,
                                                                             output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, tokenizer_eq.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(sentence, translation):
        # Translate from MWP to equation
        result, attention_weights = evaluate(sentence)

        predicted_equation = tokenizer_eq.decode([i for i in result
                                                  if i < tokenizer_eq.vocab_size])
        # Print and log the translations of the test data
        plog(f"Input: {sentence}\nHypothesis: {predicted_equation}")
        plog(f"Real Translation: {translation}")

        return predicted_equation, translation

    # Test the model's translations
    # Record the BLEU 1-gram score
    # I want to add F1 too...
    for i in range(test_text):
        # Test at most 50 problems
        predicted, actual = translate(test_text[i], test_equations[i])
        # Turn the prediction and actual translation to arrays of words
        bleu_pred = predicted.split(' ')
        bleu_act = actual.split(' ')
        score = f"BLEU 1-gram: {sentence_bleu(bleu_act, bleu_pred, weights=(1, 0, 0, 0))}"
        # Print and log the BLEU translation metric
        plog(score)

    print('...done. Script complete.')


if __name__ == "__main__":
    main()
