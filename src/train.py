"""Simplified four-model training pipeline for SmartSupport AI.
Run from project root: python -m src.train
All models share the same held-out test set. OOS recall uses a separate
hand-written OOS set and a fixed confidence threshold of 0.55.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from src.data import load_massive
from src.text_utils import normalize_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
SEED = 42
MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
MAX_LEN = 64
TRAIN_PER_INTENT = 30
OOS_THRESHOLD = 0.55
BATCH_SIZE = 4
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)


def clean(frame):
    frame = frame.copy()
    frame["text"] = frame["text"].map(normalize_text)
    return frame[frame["text"].str.len() > 0].reset_index(drop=True)


def make_oos_examples():
    return [
        "What is the capital of France?", "Explain photosynthesis to me.",
        "Who won the football match yesterday?", "Write a poem about the ocean.",
        "Can you solve this algebra homework?", "What is the population of Japan?",
        "I need advice about my university assignment.", "Translate this paragraph into Spanish.",
        "Who is the president of Brazil?", "How do I learn guitar?",
        "What is the meaning of quantum physics?", "Recommend a book about history.",
        "Help me plan a holiday in Italy.", "Write a birthday message for my friend.",
        "How do I create a website with HTML?", "Tell me about the Roman Empire.",
        "Fransa'nın başkenti neresidir?", "Fotosentezi bana açıklar mısın?",
        "Dünkü futbol maçını kim kazandı?", "Okyanus hakkında bir şiir yaz.",
        "Bu matematik ödevini çözmeme yardım et.", "Japonya'nın nüfusu kaçtır?",
        "Üniversite ödevim için tavsiyeye ihtiyacım var.", "Bu paragrafı İspanyolcaya çevir.",
        "Gitar çalmayı nasıl öğrenebilirim?", "Roma İmparatorluğu hakkında bilgi ver.",
    ]


class AverageEmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, number_of_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, number_of_classes)
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


def train_neural_network(train_df, labels):
    sentences = [text.split() for text in train_df["text"]]
    w2v = Word2Vec(sentences, vector_size=64, window=5, min_count=1,
                   workers=2, epochs=5, seed=SEED)
    vocab = {word: i + 2 for i, word in enumerate(w2v.wv.index_to_key)}
    vocab["<unk>"] = 1
    model = AverageEmbeddingNet(len(vocab) + 2, 64, len(labels))
    with torch.no_grad():
        for word, index in vocab.items():
            if word in w2v.wv:
                model.embedding.weight[index].copy_(torch.tensor(w2v.wv[word]))
    x = encode_texts(train_df["text"].tolist(), vocab)
    y = torch.tensor(train_df["label_id"].to_numpy(), dtype=torch.long)
    loader = DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        print(f"Neural network epoch {epoch + 1}/5 complete")
    model.eval()
    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "labels": labels, "embedding_dim": 64}, OUT / "nn.pt")

    def probabilities(texts):
        with torch.no_grad():
            logits = model(encode_texts([normalize_text(x) for x in texts], vocab))
            return torch.softmax(logits, dim=-1).numpy()
    return probabilities


class TicketDataset(Dataset):
    def __init__(self, texts, label_ids, tokenizer):
        self.enc = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        self.labels = list(label_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.enc.items()}
        item["labels"] = self.labels[index]
        return item


def transformer_probabilities(model, tokenizer, texts):
    model.eval()
    outputs = []
    for start in range(0, len(texts), 16):
        batch = [normalize_text(x) for x in texts[start:start + 16]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            outputs.append(torch.softmax(model(**encoded).logits, dim=-1).numpy())
    return np.concatenate(outputs, axis=0)


def artifact_size(path):
    path = Path(path)
    files = path.rglob("*") if path.is_dir() else [path]
    return sum(p.stat().st_size for p in files if p.is_file()) / (1024 * 1024)


def evaluate(name, predict_proba, test_texts, test_labels, oos_texts, artifact):
    probabilities = predict_proba(test_texts)
    predicted = probabilities.argmax(axis=1)
    oos_probabilities = predict_proba(oos_texts)
    oos_recall = float(np.mean(oos_probabilities.max(axis=1) < OOS_THRESHOLD))
    for text in test_texts[:5]:
        predict_proba([text])
    timings = []
    for i in range(100):
        start = time.perf_counter()
        predict_proba([test_texts[i % len(test_texts)]])
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "model": name,
        "accuracy": float(accuracy_score(test_labels, predicted)),
        "macro_f1": float(f1_score(test_labels, predicted, average="macro")),
        "out_of_scope_recall": oos_recall,
        "oos_confidence_threshold": OOS_THRESHOLD,
        "size_mb": artifact_size(artifact),
        "latency_ms_mean_100": float(np.mean(timings)),
        "latency_ms_p50": float(np.median(timings)),
        "hardware": f"CPU; {torch.get_num_threads()} PyTorch threads",
    }



def train_distilled_student(train_df, labels, teacher, tokenizer):
    """Train a smaller student using teacher probabilities and true labels."""
    print("\\nTraining knowledge-distilled student model...")

    sentences = [text.split() for text in train_df["text"]]
    w2v = Word2Vec(
        sentences, vector_size=64, window=5,
        min_count=1, workers=2, epochs=5, seed=SEED
    )
    vocab = {word: i + 2 for i, word in enumerate(w2v.wv.index_to_key)}
    vocab["<unk>"] = 1

    student = AverageEmbeddingNet(len(vocab) + 2, 64, len(labels))
    with torch.no_grad():
        for word, index in vocab.items():
            if word in w2v.wv:
                student.embedding.weight[index].copy_(
                    torch.tensor(w2v.wv[word])
                )

    texts = train_df["text"].tolist()
    x = encode_texts(texts, vocab)
    y = torch.tensor(train_df["label_id"].to_numpy(), dtype=torch.long)

    teacher_probs = torch.tensor(
        transformer_probabilities(teacher, tokenizer, texts),
        dtype=torch.float32
    )

    loader = DataLoader(
        torch.utils.data.TensorDataset(x, y, teacher_probs),
        batch_size=64, shuffle=True
    )

    optimizer = torch.optim.Adam(student.parameters(), lr=0.003)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    temperature = 2.0

    for epoch in range(10):
        student.train()
        for bx, by, teacher_batch in loader:
            optimizer.zero_grad()
            logits = student(bx)
            soft_loss = kl_loss(
                torch.log_softmax(logits / temperature, dim=1),
                torch.softmax(
                    torch.log(teacher_batch.clamp(min=1e-8)) / temperature,
                    dim=1
                )
            ) * (temperature ** 2)
            hard_loss = nn.CrossEntropyLoss()(logits, by)
            loss = 0.7 * soft_loss + 0.3 * hard_loss
            loss.backward()
            optimizer.step()
        print(f"Distillation epoch {epoch + 1}/10 complete")

    student.eval()
    quantized = torch.quantization.quantize_dynamic(
        student, {nn.Linear}, dtype=torch.qint8
    )
    path = OUT / "student_int8.pt"
    torch.save({
        "state_dict": quantized.state_dict(),
        "vocab": vocab,
        "labels": labels,
        "embedding_dim": 64
    }, path)

    def probabilities(texts):
        with torch.no_grad():
            logits = quantized(
                encode_texts([normalize_text(t) for t in texts], vocab)
            )
            return torch.softmax(logits, dim=-1).numpy()

    return probabilities, path


def main():
    print("Loading MASSIVE English + Turkish dataset...")
    train_df, validation_df, test_df = map(clean, load_massive())
    labels = sorted(train_df["label"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}

    train_df = (train_df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(TRAIN_PER_INTENT, len(g)),
                                          random_state=SEED))
                .reset_index(drop=True))
    for frame in (train_df, validation_df, test_df):
        frame["label_id"] = frame["label"].map(label_to_id)
        frame.dropna(subset=["label_id"], inplace=True)
        frame["label_id"] = frame["label_id"].astype(int)

    test_texts = test_df["text"].tolist()
    test_labels = test_df["label_id"].to_numpy()
    oos_texts = make_oos_examples()
    (OUT / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    print(f"Training: {len(train_df)} | Test: {len(test_df)} | Intents: {len(labels)}")
    print(f"Hand-written out-of-scope examples: {len(oos_texts)}")
    results = []

    print("\n[1/4] TF-IDF + Logistic Regression")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                 sublinear_tf=True)
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(vectorizer.fit_transform(train_df["text"]), train_df["label_id"])
    baseline_path = OUT / "baseline.joblib"
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier,
                 "labels": labels}, baseline_path)

    def baseline_probs(texts):
        normalized = [normalize_text(x) for x in texts]
        return classifier.predict_proba(vectorizer.transform(normalized))
    results.append(evaluate("TF-IDF + Logistic Regression", baseline_probs,
                            test_texts, test_labels, oos_texts, baseline_path))

    print("\n[2/4] Word2Vec + Feedforward Neural Network")
    nn_probs = train_neural_network(train_df, labels)
    results.append(evaluate("Word2Vec + Feedforward NN", nn_probs,
                            test_texts, test_labels, oos_texts, OUT / "nn.pt"))

    print("\n[3/4] Fine-tuning multilingual DistilBERT (one epoch)")
    model_name = "distilbert/distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels))
    dataset = TicketDataset(train_df["text"].tolist(),
                            train_df["label_id"].tolist(), tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=2e-5)
    transformer.train()
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        loss = transformer(**batch).loss
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"  Training batch {step}/{len(loader)}")
    transformer.eval()
    transformer_dir = OUT / "transformer_fp32"
    transformer.save_pretrained(transformer_dir)
    tokenizer.save_pretrained(transformer_dir)

    def transformer_probs(texts):
        return transformer_probabilities(transformer, tokenizer, texts)
    results.append(evaluate("Multilingual DistilBERT fine-tuned",
                            transformer_probs, test_texts, test_labels,
                            oos_texts, transformer_dir))

    print("\n[4/4] Knowledge distillation + INT8 student quantization")
    student_probs, student_path = train_distilled_student(
        train_df, labels, transformer, tokenizer
    )
    tokenizer.save_pretrained(OUT / "transformer_tokenizer")
    results.append(evaluate(
        "Distilled Word2Vec Student INT8", student_probs,
        test_texts, test_labels, oos_texts, student_path
    ))
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "results.csv", index=False)
    base_predictions = baseline_probs(test_texts).argmax(axis=1)
    report = classification_report(test_labels, base_predictions,
                                   labels=list(range(len(labels))),
                                   target_names=labels, zero_division=0)
    (OUT / "classification_report.txt").write_text(report)
    (OUT / "oos_examples.json").write_text(
        json.dumps(oos_texts, ensure_ascii=False, indent=2))
    print("\n===== RESULTS =====")
    print(result_df.to_string(index=False))
    print("\nSaved models and results to:", OUT)
    print("OOS recall threshold:", OOS_THRESHOLD)


if __name__ == "__main__":
    main()
