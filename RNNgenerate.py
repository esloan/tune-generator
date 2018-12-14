import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time
import functools

# TODO: actual data goes here
path_to_file = "ReelsNoMetadata.txt"
text = open(path_to_file).read()

vocab = sorted(set(text))

# TODO: contemplate whether I want to encode numbers differently
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
# TODO: consider what this should be: starting with ~ a tune length
seq_length = 200
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size 
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
rnn = functools.partial(
tf.keras.layers.GRU, recurrent_activation='sigmoid')

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                              batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab), 
  embedding_dim=embedding_dim, 
  rnn_units=rnn_units, 
  batch_size=BATCH_SIZE)

model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = tf.losses.sparse_softmax_cross_entropy)

  # Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 300

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

def abc_len(text):
  # Note that we're assuming measures are 4/4, note lengths are 1/8
  total = 0
  for i in range(len(text)):
    char = text[i]
    # alphabet characters are new notes: 1 beat until otherwise noted
    if char.isalpha():
      total += 0.5
    #octaves, accidentals- no added time
    modifiers = [',', '\'', '^', '=', '_']
    if char in modifiers:
      # This is a place holder, don't really need it
      total += 0

    if char.isdigit():
      total += 0.5 * (int(char) - 1)

  return total

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate

  # Converting our start string to numbers (vectorizing) 
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()

  # A part and B part
  for i in range(2):
    # Get a measure, then repeat 8 times
    # start of repeat
    text_generated.append('|')
    text_generated.append(':')
    for j in range(8):
      measure = []
      while abc_len(measure) < 4:
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        
        new_note = idx2char[predicted_id]
        # Manually check that it doesn't make the part too long 
        # or add a bar line or repeat
        bad = ['|','[',']','!',':']
        if new_note not in bad and abc_len(measure+[new_note]) <= 4:
          # We pass the predicted word as the next input to the model
          # along with the previous hidden state
          input_eval = tf.expand_dims([predicted_id], 0)
          
          text_generated.append(new_note)
          measure.append(new_note)
      # We've gotten a measure, add a bar line
      if j != 7:
        text_generated.append('|')
    # add a repeat at the end of the part
    text_generated.append(':')
    text_generated.append('|')
  return (''.join(text_generated))

for i in range(10):
  print(generate_text(model, start_string="D"))
  print("")