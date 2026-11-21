import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.text_utils import (
    normalize_text,
    is_small_talk,
    small_talk_reply,
    route_team,
    correct_product_phrases,
    extract_model_codes,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEVICE = torch.device("cpu")

st.set_page_config(page_title="SmartSupport AI", page_icon="🛠️", layout="wide")


class AverageEmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, number_of_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, number_of_classes),
        )

    def forward(self, token_ids):
        mask = (token_ids != 0).float().unsqueeze(-1)
        pooled = (self.embedding(token_ids) * mask).sum(1)
        pooled = pooled / mask.sum(1).clamp(min=1)
        return self.classifier(pooled)


def encode_texts(texts, vocabulary, max_length=48):
    array = np.zeros((len(texts), max_length), dtype=np.int64)
    for row, text in enumerate(texts):
        ids = [vocabulary.get(word, 1) for word in text.split()[:max_length]]
        array[row, :len(ids)] = ids
    return torch.tensor(array, dtype=torch.long)


@st.cache_resource
def load_artifacts():
    labels = json.loads((MODELS / "labels.json").read_text(encoding="utf-8"))

    baseline = joblib.load(MODELS / "baseline.joblib")

    nn_checkpoint = torch.load(
        MODELS / "nn.pt", map_location="cpu", weights_only=False
    )
    vocabulary = nn_checkpoint["vocab"]
    neural_net = AverageEmbeddingNet(
        max(vocabulary.values()) + 2,
        nn_checkpoint.get("embedding_dim", 64),
        len(labels),
    )
    neural_net.load_state_dict(nn_checkpoint["state_dict"])
    neural_net.eval()

    transformer_dir = MODELS / "transformer_fp32"
    tokenizer = AutoTokenizer.from_pretrained(MODELS / "transformer_tokenizer")

    transformer = AutoModelForSequenceClassification.from_pretrained(
        transformer_dir
    ).eval()

    student_checkpoint = torch.load(
        MODELS / "student_int8.pt", map_location="cpu", weights_only=False
    )
    student_vocabulary = student_checkpoint["vocab"]

    quantized_model = AverageEmbeddingNet(
        len(student_vocabulary) + 2,
        student_checkpoint["embedding_dim"],
        len(labels),
    )
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model, {nn.Linear}, dtype=torch.qint8
    )
    quantized_model.load_state_dict(student_checkpoint["state_dict"])
    quantized_model.eval()

    return (
        labels, baseline, neural_net, vocabulary, transformer,
        quantized_model, student_vocabulary, tokenizer
    )



def timed_prediction(function):
    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000


def transformer_prediction(model, tokenizer, text, labels):
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=64
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
    index = int(probabilities.argmax())
    return labels[index], float(probabilities[index])


def read_results():
    result_path = MODELS / "results.csv"
    if result_path.exists():
        return pd.read_csv(result_path)
    return None


st.title("🛠️ SmartSupport AI")
st.caption(
    "Bilingual English + Turkish intent classification and support routing"
)

required_files = [
    MODELS / "labels.json",
    MODELS / "baseline.joblib",
    MODELS / "nn.pt",
    MODELS / "transformer_fp32" / "config.json",
    MODELS / "student_int8.pt",
    MODELS / "transformer_tokenizer",
]
missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
if missing:
    st.error("Some trained model files are missing:")
    st.write(missing)
    st.stop()

try:
    labels, baseline, neural_net, vocabulary, transformer, quantized_model, student_vocabulary, tokenizer = load_artifacts()
except Exception as error:
    st.error("The trained models could not be loaded.")
    st.exception(error)
    st.stop()

st.subheader("Classify a support message")
message = st.text_area(
    "Describe your request",
    placeholder="Example: My dishwasher is not draining / Bulaşık makinem su boşaltmıyor",
    height=120,
)
threshold = st.slider(
    "Human-agent confidence threshold",
    min_value=0.30,
    max_value=0.95,
    value=0.55,
    step=0.05,
)

if st.button("Classify message", type="primary", disabled=not message.strip()):
    if is_small_talk(message):
        st.info(small_talk_reply(message))
    else:
        corrected = correct_product_phrases(message)
        normalized = normalize_text(corrected)
        model_codes = extract_model_codes(message)

        def baseline_predict():
            vector = baseline["vectorizer"].transform([normalized])
            probabilities = baseline["classifier"].predict_proba(vector)[0]
            index = int(np.argmax(probabilities))
            return labels[index], float(probabilities[index])

        def neural_predict():
            with torch.no_grad():
                logits = neural_net(encode_texts([normalized], vocabulary))
                probabilities = torch.softmax(logits, dim=-1)[0]
            index = int(probabilities.argmax())
            return labels[index], float(probabilities[index])

        def transformer_predict():
            return transformer_prediction(transformer, tokenizer, normalized, labels)

        def quantized_predict():
            with torch.no_grad():
                logits = quantized_model(
                    encode_texts([normalized], student_vocabulary)
                )
                probabilities = torch.softmax(logits, dim=-1)[0]
            index = int(probabilities.argmax())
            return labels[index], float(probabilities[index])

        predictors = [
            ("TF-IDF + Logistic Regression", baseline_predict),
            ("Word2Vec + Feedforward NN", neural_predict),
            ("DistilBERT fine-tuned", transformer_predict),
            ("Distilled Student INT8", quantized_predict),
        ]

        st.subheader("Four-model comparison")
        columns = st.columns(4)
        predictions = []
        for column, (model_name, predictor) in zip(columns, predictors):
            try:
                (intent, confidence), latency = timed_prediction(predictor)
                predictions.append((model_name, intent, confidence, latency))
                with column:
                    st.markdown(f"**{model_name}**")
                    st.metric("Confidence", f"{confidence * 100:.1f}%")
                    st.write(f"Intent: `{intent}`")
                    st.caption(f"Latency: {latency:.2f} ms")
            except Exception as error:
                predictions.append((model_name, "ERROR", 0.0, 0.0))
                with column:
                    st.error(f"{model_name} failed")
                    st.exception(error)

        st.divider()
        production = predictions[-1]
        intent, confidence = production[1], production[2]
        st.subheader("Final routing decision")
        if intent == "ERROR" or confidence < threshold:
            st.warning(
                "Route: Human Agent — the distilled student's confidence is below "
                "the selected threshold."
            )
        else:
            st.success(f"Route: {route_team(intent)}")
            st.write(f"Selected intent: `{intent}`")

        st.caption(
            "The quantized distilled student controls the final routing decision. "
            "The confidence threshold is a prototype rule; out-of-scope performance "
            "must be interpreted using the separate evaluation set."
        )

        with st.expander("Preprocessing details"):
            st.write("Corrected message:", corrected)
            st.write("Extracted product model codes:", model_codes or "None detected")
            st.write("Normalized message:")
            st.code(normalized)

st.divider()
st.subheader("Measured model evaluation")
results = read_results()
if results is not None:
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.caption(
        "Results are from the project's recorded test run. Out-of-scope recall "
        "was measured on 26 manually written examples at confidence threshold 0.55."
    )
else:
    st.info("Evaluation results will appear here after training.")
