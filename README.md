# SmartSupport AI — SWE015 Homework

A bilingual English–Turkish intent-classification and support-routing prototype developed for SWE015.

## 1. Use case

SmartSupport AI accepts free-text messages in English or Turkish, normalizes the text, predicts an intent, displays four model predictions, and routes low-confidence messages to a human agent.

**Dataset limitation:** The project uses Amazon MASSIVE, a general-purpose multilingual voice-assistant intent dataset. It does not contain real appliance-repair tickets. This project is therefore a bilingual intent-classification prototype and dataset proxy, not a model validated on real appliance-support conversations.

## 2. Four-model comparison

1. **Classical NLP baseline:** TF-IDF with unigram and bigram features + Logistic Regression.
2. **Neural network:** Word2Vec averaged word embeddings + feedforward neural-network classifier.
3. **Fine-tuned Transformer:** `distilbert/distilbert-base-multilingual-cased`, fine-tuned for one epoch.
4. **Quantized Transformer:** The fine-tuned DistilBERT model with dynamic INT8 quantization.

The application displays predictions from all four models. The quantized Transformer controls the final confidence-based routing decision. Additional preprocessing and application rules include text normalization, simple text corrections, product/model-code handling, and rule-based greeting responses.

## 3. Dataset

The project uses the English (`en-US`) and Turkish (`tr-TR`) configurations of [Amazon MASSIVE](https://huggingface.co/datasets/AmazonScience/massive).

| Partition | Combined examples |
|---|---:|
| Training | 23,028 |
| Validation | 4,066 |
| Test | 5,948 |

The dataset contains 60 intent classes. The recorded training run sampled up to 30 examples per intent, resulting in 1,776 training examples. All four models were evaluated on the full 5,948-example test partition.

A separate set of 26 manually written examples was used for an initial out-of-scope recall check. This small set is not representative of all possible out-of-scope messages.

## 4. Training and evaluation

Create and activate the virtual environment, install dependencies, train the models, and launch the app:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app/streamlit_app.py
```

The training script downloads the dataset and pretrained Transformer checkpoint. It saves model artifacts and evaluation results in `models/`. Open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`.

The recorded evaluation includes accuracy, macro-F1, out-of-scope recall, model size, mean latency over 100 single-message runs, p50 latency, and hardware. The confidence threshold used for the recorded out-of-scope check and default routing is 0.55.

## 5. Recorded evaluation results

The following values are from the actual recorded training run on CPU with four PyTorch threads.

| Model | Accuracy | Macro-F1 | OOS recall | Mean latency (ms) |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.5456 | 0.5160 | 1.0000 | 2.2523 |
| Word2Vec + Feedforward NN | 0.2927 | 0.3345 | 1.0000 | 0.3910 |
| Multilingual DistilBERT fine-tuned | 0.2712 | 0.2185 | 1.0000 | 33.8144 |
| Multilingual DistilBERT INT8 | 0.2645 | 0.2241 | 1.0000 | 34.5229 |

**Interpretation:** TF-IDF + Logistic Regression achieved the highest accuracy and macro-F1 in this experiment. INT8 was slightly less accurate and slightly slower than the unquantized Transformer in the recorded latency test. Do not describe quantization as a speed improvement for these results.

The out-of-scope recall of 1.0 was measured on only 26 manually written examples at threshold 0.55. It does not establish robust real-world out-of-scope detection. Low confidence indicates uncertainty; it does not by itself prove that a message is out of scope.

Model-size values are available in `models/results.csv`. Check the size-measurement method before making direct comparisons, because saved artifacts may include different files.

## 6. Application behavior

1. Enter a message in English or Turkish.
2. The application preprocesses and normalizes the text.
3. Four models produce intent predictions and confidence scores.
4. The application displays the predictions side by side.
5. The quantized INT8 model supplies the final prediction.
6. If its confidence is below the selected threshold, the message is routed to a human agent. Otherwise, the predicted intent is used for routing.
7. Greetings and simple small talk are handled using application rules.

## 7. Repository structure

```text
app/
    streamlit_app.py
src/
    data.py
    data_original.py
    text_utils.py
    train.py
    train_original.py
    train_before_simplification.py
models/
    baseline.joblib
    nn.pt
    labels.json
    results.csv
    classification_report.txt
    oos_examples.json
    transformer_fp32/
    transformer_tokenizer/
    transformer_int8.pt
requirements.txt
README.md
```

Generated model artifacts are stored locally in `models/`. Large model weights should not be committed to a public Git repository. If sharing the repository without these files, provide instructions for generating or downloading them.

## 8. Limitations and future improvements

- MASSIVE is a general voice-assistant dataset, not a real appliance-support ticket dataset.
- The training run uses a limited sample of the available training data.
- The manually written out-of-scope set contains only 26 examples.
- The confidence threshold is a simple prototype rule and requires calibration on representative support data.
- The models can disagree, especially on Turkish or unfamiliar messages.
- Future work should collect and label real bilingual appliance-support messages, evaluate on a domain-specific held-out test set, and improve confidence calibration and out-of-scope detection.

## 9. AI usage statement

ChatGPT was used to assist with drafting and debugging parts of the project code, documentation, and presentation materials. The student is responsible for running and testing the code, understanding the implementation, verifying the reported results, and disclosing AI assistance in accordance with course requirements.

## 10. Reference

Amazon Science. *MASSIVE: A 1 Million-Example Multilingual Natural Language Understanding Dataset with 51 Languages.*

https://huggingface.co/datasets/AmazonScience/massive

## Trained Model Weights

The trained model files required to run SmartSupport AI are available here:

**[Download SmartSupport AI Model Weights](https://drive.google.com/drive/folders/1TcwZARe6cLBpz3SEpYjUjGU8GWl4Klg3?usp=sharing)**

The folder contains:
- TF-IDF + Logistic Regression model (`baseline.joblib`)
- Word2Vec + Feedforward Neural Network (`nn.pt`)
- Fine-tuned multilingual DistilBERT model (`transformer_fp32/`)
- Quantized INT8 model (`transformer_int8.pt`)
- Transformer tokenizer (`transformer_tokenizer/`)

Download the model files and place them in the project's `models/` directory before running the application.
