import math
import streamlit as st
import spacy
import torch
import torch.nn as nn

torch.manual_seed(42)


st.set_page_config(page_title="Transformer Translator", layout="wide")
st.title("Transformer English → Dutch Translator")


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

english_corpus = [
    "I love mathematics",
    "calculus is great",
    "mathematics is useful",
    "I love calculus",
    "mathematics is great",
]

dutch_corpus = [
    "ik hou van wiskunde",
    "calculus is geweldig",
    "wiskunde is nuttig",
    "ik hou van calculus",
    "wiskunde is geweldig",
]

SPECIAL = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

def build_vocab(sentences):
    w2i = {t: i for i, t in enumerate(SPECIAL)}
    i2w = {i: t for i, t in enumerate(SPECIAL)}

    idx = len(SPECIAL)
    for s in sentences:
        for tok in nlp(s):
            w = tok.text.lower()
            if w not in w2i:
                w2i[w] = idx
                i2w[idx] = w
                idx += 1
    return w2i, i2w

def encode_src(s, src_w2i):
    return [src_w2i.get(t.text.lower(), src_w2i["<UNK>"]) for t in nlp(s)]

def encode_tgt(s, tgt_w2i):
    ids = [tgt_w2i["<SOS>"]]
    ids.extend([tgt_w2i.get(t.text.lower(), tgt_w2i["<UNK>"]) for t in nlp(s)])
    ids.append(tgt_w2i["<EOS>"])
    return ids

def pad(seq, l, pad=0):
    return seq + [pad] * (l - len(seq))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.h = heads
        self.d = d_model // heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.q(q).view(B, -1, self.h, self.d).transpose(1, 2)
        K = self.k(k).view(B, -1, self.h, self.d).transpose(1, 2)
        V = self.v(v).view(B, -1, self.h, self.d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, -1)
        ctx = torch.matmul(attn, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, -1, self.h * self.d)
        return self.out(ctx)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.n1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.n1(x + self.attn(x, x, x))
        x = self.n2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.n1 = nn.LayerNorm(d_model)

        self.cross = MultiHeadAttention(d_model, heads)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, x, enc, mask):
        x = self.n1(x + self.self_attn(x, x, x, mask))
        x = self.n2(x + self.cross(x, enc, enc))
        x = self.n3(x + self.ff(x))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, heads=4):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)

        self.pos = PositionalEncoding(d_model)
        self.encoder = EncoderLayer(d_model, heads)
        self.decoder = DecoderLayer(d_model, heads)
        self.fc = nn.Linear(d_model, tgt_vocab)

    def causal_mask(self, n):
        m = torch.tril(torch.ones(n, n))
        return m.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt):
        src = self.pos(self.src_emb(src))
        enc = self.encoder(src)

        tgt = self.pos(self.tgt_emb(tgt))
        mask = self.causal_mask(tgt.size(1))
        dec = self.decoder(tgt, enc, mask)
        return self.fc(dec)

@st.cache_resource
def build_and_train_model():
    src_w2i, src_i2w = build_vocab(english_corpus)
    tgt_w2i, tgt_i2w = build_vocab(dutch_corpus)

    src = [encode_src(s, src_w2i) for s in english_corpus]
    tgt = [encode_tgt(s, tgt_w2i) for s in dutch_corpus]

    src_len = max(map(len, src))
    tgt_len = max(map(len, tgt))

    src_tensor = torch.tensor([pad(x, src_len) for x in src])
    tgt_tensor = torch.tensor([pad(x, tgt_len) for x in tgt])

    decoder_input = tgt_tensor[:, :-1]
    target = tgt_tensor[:, 1:]

    model = Transformer(len(src_w2i), len(tgt_w2i))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_w2i["<PAD>"])
    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    progress_bar = st.progress(0, text="Training transformer...")
    status_text = st.empty()

    for epoch in range(100):
        opt.zero_grad()
        out = model(src_tensor, decoder_input)
        loss = loss_fn(out.reshape(-1, len(tgt_w2i)), target.reshape(-1))
        loss.backward()
        opt.step()

        if (epoch + 1) % 50 == 0:
            progress_bar.progress((epoch + 1) / 100, text=f"Training transformer... epoch {epoch + 1}/100")
            status_text.write(f"Epoch {epoch + 1}/100 - Loss: {loss.item():.6f}")

    return {
        "model": model,
        "src_w2i": src_w2i,
        "src_i2w": src_i2w,
        "tgt_w2i": tgt_w2i,
        "tgt_i2w": tgt_i2w,
        "src_len": src_len,
        "tgt_len": tgt_len,
    }

data = build_and_train_model()
model = data["model"]
src_w2i = data["src_w2i"]
tgt_w2i = data["tgt_w2i"]
src_i2w = data["src_i2w"]
tgt_i2w = data["tgt_i2w"]
src_len = data["src_len"]
tgt_len = data["tgt_len"]

def translate(sentence):
    model.eval()
    s = torch.tensor([pad(encode_src(sentence, src_w2i), src_len)])
    generated = [tgt_w2i["<SOS>"]]

    with torch.no_grad():
        for _ in range(tgt_len):
            dec = torch.tensor([generated])
            out = model(s, dec)
            nxt = out[0, -1].argmax().item()
            if nxt == tgt_w2i["<EOS>"]:
                break
            generated.append(nxt)

    words = [tgt_i2w[i] for i in generated[1:]]
    return " ".join(words)

with st.sidebar:
    st.subheader("Examples")
    for en, nl in zip(english_corpus, dutch_corpus):
        st.write(f"{en} → {nl}")

st.write("Type an English sentence and the app will translate it to Dutch.")

user_input = st.text_input("English sentence", "I love mathematics")

if st.button("Translate"):
    result = translate(user_input)
    st.success(f"Translation: {result}")

st.subheader("Built-in examples")
for sentence in english_corpus:
    st.write(f"{sentence} → {translate(sentence)}")