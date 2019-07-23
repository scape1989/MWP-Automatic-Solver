from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
# The Transformer model
from models.transformer.MultiHeadAttention import MultiHeadAttention
from models.transformer.EncoderLayer import EncoderLayer
from models.transformer.Encoder import Encoder
from models.transformer.DecoderLayer import DecoderLayer
from models.transformer.Decoder import Decoder
from models.transformer.Transformer import Transformer
from models.transformer.CustomSchedule import CustomSchedule
from models.transformer.network import create_masks, loss_function
from data.util.utils import load_data_from_binary, to_binary, log, plog, get_as_tuple, expressionize, print_epoch
from data.util.NumberTag import NumberTag

# Utilities
import time
import numpy
import os
import sys
import json
import random

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

if not len(sys.argv) > 1:
    raise Exception("Please use a config file.")

with open(os.path.join(DIR_PATH, sys.argv[1]), encoding='utf-8-sig') as fh:
    data = json.load(fh)

settings = dict(data)

DATASET = settings["dataset"]
TEST_SET = settings["test"]
DATA_PATH = os.path.join(DIR_PATH, "data/" + DATASET)
EQUALS_SIGN = False

# If fine-tuning set this to a str containing the model name
CKPT_MODEL = settings["model"]

# SubwordTextEncoder / TextEncoder
# Will get W2V, GloVe, maybe BERT in if this does poorly
ENCODE_METHOD = tfds.features.text.SubwordTextEncoder
TRAIN_WITH_TAGS = True

# Data constraints
MAX_LENGTH = 40

#|##### TF EXAMPLE ######
#  num_layers = 4
#  d_model = 128
#  dff = 512
#  num_heads = 8
#|#######################
# Hyperparameters
NUM_LAYERS = settings["layers"]
D_MODEL = settings["d_model"]
DFF = settings["dff"]
NUM_HEADS = settings["heads"]
DROPOUT = settings["dropout"]
LEARNING_RATE = settings["lr"]

# Training settings
EPOCHS = settings["epochs"]
BATCH_SIZE = settings["batch"]

# Adam optimizer params
BETA_1 = settings["beta_1"]
BETA_2 = settings["beta_2"]
EPSILON = 1e-9

# Random seed for shuffling the data
SEED = settings["seed"]

if isinstance(CKPT_MODEL, str):
    # If a model name is given train from that model
    CONTINUE_FROM_CKPT = True
else:
    CONTINUE_FROM_CKPT = False

# The name to keep track of any changes
if not CONTINUE_FROM_CKPT:
    MODEL_NAME = f"t_{NUM_LAYERS}_{NUM_HEADS}_{D_MODEL}_{DFF}_{int(time.time())}"
else:
    MODEL_NAME = CKPT_MODEL
    CHECKPOINT_PATH = os.path.join(DIR_PATH,
                                   f"models/trained/{CKPT_MODEL}/")

# The checkpoint file where the trained weights will be saved
# Only saves on finish
if not os.path.isdir(f"models/trained/{MODEL_NAME}"):
    os.mkdir(f"models/trained/{MODEL_NAME}")

MODEL_PATH = os.path.join(DIR_PATH,
                          f"models/trained/{MODEL_NAME}/")

ARE_TOKENIZERS_PRESENT = os.path.exists(os.path.join(DIR_PATH, f"models/tokenizers/{MODEL_NAME}_t.pickle")) \
    or os.path.exists(os.path.join(DIR_PATH,
                                   f"models/tokenizers/{MODEL_NAME}_e.pickle"))

# Set the seed for random
random.seed(SEED)


def main():
    print("Starting the Math Word Problem (MWP) Transformer training.")
    print(f"\nTokenizing data from {DATASET}...\n")

    examples = load_data_from_binary(DATA_PATH)

    print(f"Shuffling data with seed: {SEED}")
    random.shuffle(examples)

    train_text = []
    train_equations = []

    # Get training examples
    for example in examples:
        try:
            if not TRAIN_WITH_TAGS:
                txt, exp = get_as_tuple(example)

                if not EQUALS_SIGN:
                    train_equations.append(expressionize(exp))
                else:
                    train_equations.append(exp)

                train_text.append(txt)
            else:
                txt, exp = get_as_tuple(example)
                masked_txt, masked_exp = NumberTag(txt, exp).get_masked()

                if not EQUALS_SIGN:
                    train_equations.append(expressionize(masked_exp))
                else:
                    train_equations.append(masked_exp)

                train_text.append(masked_txt)
        except:
            # Some of the filtered questions are empty... Working on a fix
            pass

        # Convert arrays to TensorFlow constants
        train_text_const = tf.constant(train_text)
        train_eq_const = tf.constant(train_equations)

        # Turn the constants into TensorFlow Datasets
        t_dataset = tf.data.Dataset.from_tensor_slices((train_text_const,
                                                        train_eq_const))

    print("Building vocabulary...")

    if not ARE_TOKENIZERS_PRESENT:
        # Create data tokenizers
        tokenizer_txt = ENCODE_METHOD.build_from_corpus((txt.numpy() for txt, eq in t_dataset),
                                                        target_vocab_size=2**15)

        tokenizer_eq = ENCODE_METHOD.build_from_corpus((eq.numpy() for txt, eq in t_dataset),
                                                       target_vocab_size=2**15)

        to_binary(os.path.join(DIR_PATH, f"models/tokenizers/{MODEL_NAME}_t.pickle"),
                  tokenizer_txt)
        to_binary(os.path.join(DIR_PATH, f"models/tokenizers/{MODEL_NAME}_e.pickle"),
                  tokenizer_eq)
    else:
        tokenizer_txt = load_data_from_binary(
            f"models/tokenizers/{MODEL_NAME}_t.pickle")
        tokenizer_eq = load_data_from_binary(
            f"models/tokenizers/{MODEL_NAME}_e.pickle")

        print("Loaded tokenizers from file.")

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

    input_vocab_size = tokenizer_txt.vocab_size + 2
    target_vocab_size = tokenizer_eq.vocab_size + 2

    print("...done.")
    print("\nDefining the Transformer model...")

    # Using the Adam optimizer
    # Can also use CustomSchedule(D_MODEL) for lr
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE,
                                         beta_1=BETA_1,
                                         beta_2=BETA_2,
                                         epsilon=EPSILON)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_acc")

    transformer = Transformer(NUM_LAYERS,
                              D_MODEL,
                              NUM_HEADS,
                              DFF,
                              input_vocab_size,
                              target_vocab_size,
                              DROPOUT)

    print("...done.")
    print("\nTraining...")

    # Model saving
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    if CONTINUE_FROM_CKPT:
        # Load last checkpoint
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  CHECKPOINT_PATH,
                                                  max_to_keep=999)
    else:
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  MODEL_PATH,
                                                  max_to_keep=999)

    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint and CONTINUE_FROM_CKPT:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"Restored from {CHECKPOINT_PATH} checkpoint")

     # Log all the settings used in the session
    log(MODEL_NAME, MODEL_NAME)
    log(MODEL_PATH, MODEL_NAME)
    if CONTINUE_FROM_CKPT:
        log(f"Continued from {CHECKPOINT_PATH}", MODEL_NAME)
    log(DATASET, MODEL_NAME)
    log(f"Random Shuffle Seed: {SEED}", MODEL_NAME)
    log(f"\nEpochs: {EPOCHS}", MODEL_NAME)
    log(f"Batch Size: {BATCH_SIZE}", MODEL_NAME)
    log(f"Max Length: {MAX_LENGTH}", MODEL_NAME)
    log(f"Equals Sign: {EQUALS_SIGN}", MODEL_NAME)
    log(f"Layers: {NUM_LAYERS}", MODEL_NAME)
    log(f"Heads: {NUM_HEADS}", MODEL_NAME)
    log(f"Model Depth: {D_MODEL}", MODEL_NAME)
    log(f"Feed Forward Depth: {DFF}", MODEL_NAME)
    log(f"Dropout: {DROPOUT}\n", MODEL_NAME)
    log(f"Learning Rate: {LEARNING_RATE}\n", MODEL_NAME)
    log(f"Adam Params: b1 {BETA_1} b2 {BETA_2} e {EPSILON}\n", MODEL_NAME)

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

    # Train
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_acc.reset_states()

        # inp -> MWP, tar -> Equation
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 10 == 0:
                print_epoch("Epoch {}/{} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1,
                    EPOCHS,
                    batch,
                    train_loss.result(),
                    train_acc.result()))

        if epoch == (EPOCHS - 1):
            # Save a checkpoint of model weights
            ckpt_save_path = ckpt_manager.save()
            print(f'Saved {MODEL_NAME} to {ckpt_save_path}\n')
            os.remove(os.path.join(DIR_PATH, sys.argv[1]))

            settings["model"] = MODEL_NAME

            with open(os.path.join(DIR_PATH, sys.argv[1]), mode="w") as fh:
                json.dump(settings, fh)

        print_epoch("Epoch {}/{} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
            epoch + 1,
            EPOCHS,
            batch,
            train_loss.result(),
            train_acc.result()), clear=True)
        # Save a log of the epoch results
        log(f"Epoch {epoch + 1}: loss {train_loss.result()} acc {train_acc.result()}", MODEL_NAME)

        # Calculate the time the epoch took to complete
        # The first epoch seems to take significantly longer than the others
        print(f"Epoch took {int(time.time() - start)}s\n")

    print("...done.")

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
        plog(f"Input: {sentence}\nHypothesis: {predicted_equation}", MODEL_NAME)
        plog(f"Real Translation: {translation}", MODEL_NAME)

        return predicted_equation, translation

    if isinstance(TEST_SET, str):
        print(f'\nTesting translations...')

        if TEST_SET == "infix":
            sets = ["test_ai_infix.pickle",
                    "test_cc_infix.pickle", "test_il_infix.pickle"]
        elif TEST_SET == "postfix":
            sets = ["test_ai_postfix.pickle",
                    "test_cc_postfix.pickle", "test_il_postfix.pickle"]
        elif TEST_SET == "postfix":
            sets = ["test_ai_prefix.pickle",
                    "test_cc_prefix.pickle", "test_il_prefix.pickle"]

        for s in sets:
            log(s, MODEL_NAME)

            test_set = load_data_from_binary(
                os.path.join(DIR_PATH, "data/" + s))
            # Test the model's translations on withheld data
            for i, data in enumerate(test_set):
                data_dict = dict(data)
                predicted, actual = translate(data_dict["question"],
                                              expressionize(data_dict["equation"]))
        print("...done.")
    print("Exiting.")


if __name__ == "__main__":
    main()
