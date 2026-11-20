from datasets import load_dataset, concatenate_datasets

LOCALES = ["en-US", "tr-TR"]


def load_massive():
    """Load English and Turkish MASSIVE splits and make stable string labels."""
    train_parts, valid_parts, test_parts = [], [], []
    for locale in LOCALES:
        ds = load_dataset("AmazonScience/massive", locale, trust_remote_code=True)
        # MASSIVE exposes utterances as `utt` and intent as a ClassLabel.
        features = ds["train"].features
        intent_feature = features["intent"]
        def convert(example):
            label = example["intent"]
            if hasattr(intent_feature, "int2str"):
                label = intent_feature.int2str(label)
            return {"text": example.get("utt", example.get("utterance", "")), "label": str(label), "locale": locale}
        for split_name, bucket in (("train", train_parts), ("validation", valid_parts), ("test", test_parts)):
            if split_name not in ds:
                continue
            mapped = ds[split_name].map(convert, remove_columns=ds[split_name].column_names)
            bucket.append(mapped)
    return tuple(concatenate_datasets(x).to_pandas() for x in (train_parts, valid_parts, test_parts))
