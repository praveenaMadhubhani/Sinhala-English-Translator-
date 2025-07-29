from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import AdditiveAttention

app = Flask(__name__)
enc_se = load_model("enc_sin_en.h5", compile=False, custom_objects={"AdditiveAttention": AdditiveAttention})
dec_se = load_model("dec_sin_en.h5", compile=False, custom_objects={"AdditiveAttention": AdditiveAttention})
enc_es = load_model("enc_en_sin.h5", compile=False, custom_objects={"AdditiveAttention": AdditiveAttention})
dec_es = load_model("dec_en_sin.h5", compile=False, custom_objects={"AdditiveAttention": AdditiveAttention})

with open("tokenizer_sin_en.pkl", "rb") as f:
    tok_se_src, tok_se_tgt, max_len_src_se, max_len_tgt_se = pickle.load(f)
with open("tokenizer_en_sin.pkl", "rb") as f:
    tok_es_src, tok_es_tgt, max_len_src_es, max_len_tgt_es = pickle.load(f)

def predict(sentence, src_tok, tgt_tok, enc_model, dec_model, max_len_src, max_len_tgt):
    seq = src_tok.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_len_src, padding='post')
    enc_out, h, c = enc_model.predict(seq)
    tgt_seq = np.array([[tgt_tok.word_index.get('<start>')]])
    output = ''
    for _ in range(max_len_tgt):
        preds, h, c = dec_model.predict([tgt_seq, enc_out, h, c])
        idx = np.argmax(preds[0, 0, :])
        word = tgt_tok.index_word.get(idx)
        if word == '<end>' or word is None:
            break
        output += word + ' '
        tgt_seq = np.array([[idx]])
    return output.strip()

def detect_lang(text):
    latin = sum(c.isascii() and c.isalpha() for c in text)
    return "en" if latin / len(text) > 0.6 else "si"

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST":
        text = request.form["text"]
        lang = detect_lang(text)
        if lang == "si":
            translation = predict(text, tok_se_src, tok_se_tgt, enc_se, dec_se, max_len_src_se, max_len_tgt_se)
        else:
            translation = predict(text, tok_es_src, tok_es_tgt, enc_es, dec_es, max_len_src_es, max_len_tgt_es)
    return render_template("index.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)
