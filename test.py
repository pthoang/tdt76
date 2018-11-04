import tensorflow as tf
from tensorflow import keras
num_decoder_tokens, latent_dim = (10, 10)
num_encoder_tokens = 10
encoder_inputs = keras.layers.Input(shape=(None,))
x_1 = keras.layers.Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
x, state_h, state_c = keras.layers.LSTM(latent_dim,
                           return_state=True)(x_1)
print(x_1)
x_1 = keras.layers.Lambda(lambda input: tf.reshape(input, (1,10)))(x_1)
print(x_1)

encoder_states = [x_1, state_c]


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.layers.Input(shape=(None,))
x = keras.layers.Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x = keras.layers.LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = keras.layers.Dense(num_decoder_tokens, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

print(model.summary())
for layer in model.layers:
    print(layer)

