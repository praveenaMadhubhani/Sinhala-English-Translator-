from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, AdditiveAttention

def create_models(vocab_inp, vocab_tgt, latent_dim=256):
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(vocab_inp, latent_dim)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    dec_emb = Embedding(vocab_tgt, latent_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    attention = AdditiveAttention()
    attention_out = attention([decoder_outputs, enc_emb])
    decoder_concat_input = Dense(latent_dim, activation='tanh')(attention_out)
    decoder_dense = Dense(vocab_tgt, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    encoder_model = Model(encoder_inputs, [enc_emb, state_h, state_c])

    dec_state_input_h = Input(shape=(latent_dim,))
    dec_state_input_c = Input(shape=(latent_dim,))
    enc_output_inf = Input(shape=(None, latent_dim))
    dec_inputs = Input(shape=(None,))
    dec_emb2 = Embedding(vocab_tgt, latent_dim)(dec_inputs)
    dec_outputs, state_h_inf, state_c_inf = decoder_lstm(dec_emb2, initial_state=[dec_state_input_h, dec_state_input_c])
    attn_out_inf = attention([dec_outputs, enc_output_inf])
    dec_concat_inf = Dense(latent_dim, activation='tanh')(attn_out_inf)
    dec_out_inf = decoder_dense(dec_concat_inf)
    decoder_model = Model([dec_inputs, enc_output_inf, dec_state_input_h, dec_state_input_c],
                          [dec_out_inf, state_h_inf, state_c_inf])
    return model, encoder_model, decoder_model
