import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import create_models

df = pd.read_csv("converted_data.csv")[['Sinhala', 'English']].dropna()
df['English'] = df['English'].apply(lambda x: "<start> " + x + " <end>")
df['Sinhala'] = df['Sinhala'].apply(lambda x: "<start> " + x + " <end>")

# Sinhala → English
src_tokenizer_se = Tokenizer(filters='')
src_tokenizer_se.fit_on_texts(df['Sinhala'])
src_seq_se = src_tokenizer_se.texts_to_sequences(df['Sinhala'])
max_src_len_se = max(len(seq) for seq in src_seq_se)
src_pad_se = pad_sequences(src_seq_se, maxlen=max_src_len_se, padding='post')

tgt_tokenizer_se = Tokenizer(filters='')
tgt_tokenizer_se.fit_on_texts(df['English'])
tgt_seq_se = tgt_tokenizer_se.texts_to_sequences(df['English'])
max_tgt_len_se = max(len(seq) for seq in tgt_seq_se)
tgt_pad_se = pad_sequences(tgt_seq_se, maxlen=max_tgt_len_se, padding='post')

decoder_input_se = tgt_pad_se[:, :-1]
decoder_output_se = tgt_pad_se[:, 1:]

vocab_src_se = len(src_tokenizer_se.word_index) + 1
vocab_tgt_se = len(tgt_tokenizer_se.word_index) + 1

model_se, enc_se, dec_se = create_models(vocab_src_se, vocab_tgt_se)
model_se.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_se.fit([src_pad_se, decoder_input_se], decoder_output_se, batch_size=64, epochs=10, validation_split=0.1)

enc_se.save("enc_sin_en.h5")
dec_se.save("dec_sin_en.h5")
with open("tokenizer_sin_en.pkl", "wb") as f:
    pickle.dump((src_tokenizer_se, tgt_tokenizer_se, max_src_len_se, max_tgt_len_se), f)

# English → Sinhala
src_tokenizer_es = tgt_tokenizer_se
tgt_tokenizer_es = src_tokenizer_se
src_seq_es = tgt_seq_se
tgt_seq_es = src_seq_se
max_src_len_es = max_tgt_len_se
max_tgt_len_es = max_src_len_se
src_pad_es = tgt_pad_se
tgt_pad_es = src_pad_se

decoder_input_es = tgt_pad_es[:, :-1]
decoder_output_es = tgt_pad_es[:, 1:]

vocab_src_es = vocab_tgt_se
vocab_tgt_es = vocab_src_se

model_es, enc_es, dec_es = create_models(vocab_src_es, vocab_tgt_es)
model_es.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_es.fit([src_pad_es, decoder_input_es], decoder_output_es, batch_size=64, epochs=10, validation_split=0.1)

enc_es.save("enc_en_sin.h5")
dec_es.save("dec_en_sin.h5")
with open("tokenizer_en_sin.pkl", "wb") as f:
    pickle.dump((src_tokenizer_es, tgt_tokenizer_es, max_src_len_es, max_tgt_len_es), f)

print("✅ Training completed and models saved.")
