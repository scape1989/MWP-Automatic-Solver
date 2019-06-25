from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

from models.transformer.MultiHeadAttention import MultiHeadAttention
from models.transformer.EncoderLayer import EncoderLayer
from models.transformer.Encoder import Encoder
from models.transformer.DecoderLayer import DecoderLayer
from models.transformer.Decoder import Decoder
from models.transformer.Transformer import Transformer
from models.transformer.CustomSchedule import CustomSchedule
from models.transformer.network import create_masks

import time
import numpy as np
import os
import re
import pickle

tf.enable_eager_execution()

DIR_PATH = DIR_PATH = os.path.abspath(os.path.dirname(__file__))
DATASET = 'data/postfix_data.pickle'
DATA_PATH = os.path.join(DIR_PATH, DATASET)
VALIDATION_SPLIT = 0.1
EQUALS_SIGN = False

# Data constraints
BUFFER_SIZE = 400000
BATCH_SIZE = 32
MAX_LENGTH = 40

# Hyperparameters
NUM_LAYERS = 4
D_MODEL = 256
DFF = 256
NUM_HEADS = 8
DROPOUT = 0.05

EPOCHS = 20

# Adam params
BETA_1 = 0.9
BETA_2 = 0.98
EPSILON = 1e-9

# The name to keep track of any changes
MODEL_NAME = f"t{EPOCHS}_{NUM_LAYERS}_{D_MODEL}_{DFF}_{NUM_HEADS}_{BATCH_SIZE}_{int(time.time())}"


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


def main():
    print('Starting the Math Word Problem (MWP) Transformer training.')
    print(f'\nTokenizing data from {DATASET}...\n')

    log(DATASET)
    log(f"\nEpochs: {EPOCHS}")
    log(f"Max Length: {MAX_LENGTH}")
    log(f"Equals Sign: {EQUALS_SIGN}")
    log(f"Layers: {NUM_LAYERS}")
    log(f"Heads: {NUM_HEADS}")
    log(f"DModel: {D_MODEL}")
    log(f"DFF: {DFF}")
    log(f"Dropout: {DROPOUT}")

    examples = read_data_from_file(DATA_PATH)

    train_text = []
    train_equations = []

    val_text = []
    val_equations = []

    split_point = int((1 - VALIDATION_SPLIT) * len(examples))

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
    log(f"Data split <- {len(train_text)}:{len(val_text)}")

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

    tokenizer_txt = tfds.features.text.TextEncoder.build_from_corpus(
        (txt.numpy() for txt, eq in t_dataset), target_vocab_size=2**13)

    tokenizer_eq = tfds.features.text.TextEncoder.build_from_corpus(
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

    val_dataset = v_dataset.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE,
                                                                     padded_shapes=([-1], [-1]))

    input_vocab_size = tokenizer_txt.vocab_size + 2
    target_vocab_size = tokenizer_eq.vocab_size + 2

    print('...done.')
    print('\nDefining the Transformer model...')

    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=BETA_1,
                                         beta_2=BETA_2,
                                         epsilon=EPSILON)

    temp_learning_rate_schedule = CustomSchedule(D_MODEL)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                              reduction='none')(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    #true_pos = tf.keras.metrics.TruePositives(name='true_positive')
    #false_pos = tf.keras.metrics.FalsePositives(name='false_positive')
    #false_neg = tf.keras.metrics.FalseNegatives(name='false_negative')

    def print_epoch(epoch, batch):
        # true_pos.reset_states()
        # false_pos.reset_states()
        # false_neg.reset_states()

        # t_p = true_pos.result()
        # f_p = false_pos.result()
        # f_n = false_neg.result()

        # Recall = TP/TP+FN
        # recall = t_p / (t_p + f_n)
        # Precision = TP/TP+FP
        # precision = t_p / (t_p + f_p)
        # F1 Score = 2*(Recall * Precision) / (Recall + Precision)
        # f1 = 2 * (recall * precision) / (recall + precision)

        epoch_information = 'Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, batch, train_loss.result(), train_accuracy.result())

        print(epoch_information)

        # Save a log of the epoch results
        log(epoch_information)

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
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)

            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

        # true_pos.update_state(tar_real, predictions)
        # false_pos.update_state(tar_real, predictions)
        # false_neg.update_state(tar_real, predictions)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> MWP, tar -> Equation
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 500 == 0:
                print_epoch(epoch, batch)

        if epoch == (EPOCHS - 1):
            transformer.save_weights(
                f"models/trained/{MODEL_NAME}{int(train_accuracy.result())}.h5")
            print(f'Saved {MODEL_NAME}\n')

        print_epoch(epoch, batch)

        print(f'The last epoch took {int(time.time() - start)}s\n')

    print('...done.')
    print(f'\nTesting translations after {EPOCHS} epochs...')

    def evaluate(inp_sentence):
        start_token = [tokenizer_txt.vocab_size]
        end_token = [tokenizer_txt.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + \
            tokenizer_txt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [tokenizer_eq.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

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

    def translate(sentence):
        result, attention_weights = evaluate(sentence)

        predicted_equation = tokenizer_eq.decode([i for i in result
                                                  if i < tokenizer_eq.vocab_size])

        translation = f'Input: {sentence}\nPredicted translation: {predicted_equation}'
        print(translation)
        log(translation)

    translate(
        "george has 20 bikes. he sells 1/2 of them for $ 85 each. how much money did george make?")
    real = "Real translation: (20 / 2) * 85"
    print(real)
    log(real)

    translate("manny writes 100 letters to his grandma. 46 letters were handwritten. how many letters were not hand written?")
    real = "Real translation: 100 - 46"
    print(real)
    log(real)

    translate("a store sells 20 percent of its inventory of disco balls to a local hippie. if they started with 963332 disco balls, how many do they have left?")
    real = "Real translation: 0.2 * 963332"
    print(real)
    log(real)

    translate("jim and pam each have 4 pencils and 12 pens. if jim gives pam 75 percent of his pens, how many pens does pam have now?")
    real = "Real translation: 12 + 8"
    print(real)
    log(real)

    translate("darth vader collected 69 lightsabers. skywalker collected 2 times as many lightsabers. how many lightsabers did skywalker collect?")
    real = "Real translation: 69 * 2"
    print(real)
    log(real)

    translate("what is 6 divided by 200?")
    real = "Real translation: 6 / 200"
    print(real)
    log(real)

    translate("what is 12 times 12?")
    real = "Real translation: 12 * 12"
    print(real)
    log(real)

    translate("calculate 80 percent of 283")
    real = "Real translation: (80 / 100) * 283"
    print(real)
    log(real)

    translate("add 79 to 21898")
    real = "Real translation: 79 + 21898"
    print(real)
    log(real)

    translate("subtract 0.212 from 42")
    real = "Real translation: 42 - 0.212"
    print(real)
    log(real)

    print('...done. Script complete.')


if __name__ == "__main__":
    main()
