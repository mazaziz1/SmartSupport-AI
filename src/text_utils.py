import re
import unicodedata
from difflib import get_close_matches

GREETINGS = {"hi", "hello", "hey", "merhaba", "selam", "good morning", "good evening"}
PRODUCT_PHRASES = ["dishwasher", "washing machine", "refrigerator", "fridge", "oven", "bulaşık makinesi", "çamaşır makinesi", "buzdolabı", "fırın"]


def normalize_text(text: str) -> str:
    """Normalize noisy Turkish/English support text while retaining letters."""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # coooool -> cool
    text = re.sub(r"\b[0-9]{2,}\b", " MODELNUMBER ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_model_codes(text: str) -> list[str]:
    """Extract appliance-style model codes such as DW-6534 or WM1234."""
    return re.findall(r"\b[A-Za-z]{1,5}[- ]?\d{3,8}\b", str(text))


def correct_product_phrases(text: str) -> str:
    """Correct close misspellings of a small, explicit product lexicon."""
    result = str(text)
    for candidate in PRODUCT_PHRASES:
        # Match one- or two-word spans only; conservative threshold avoids broad rewrites.
        words = candidate.split()
        pattern = r"\b[\w]+(?:\s+[\w]+)?\b" if len(words) == 2 else r"\b[\w]+\b"
        for match in list(re.finditer(pattern, result, flags=re.IGNORECASE)):
            token = match.group(0)
            if token.lower() == candidate.lower():
                continue
            close = get_close_matches(token.lower(), [candidate.lower()], n=1, cutoff=0.86)
            if close:
                result = result[:match.start()] + candidate + result[match.end():]
    return result


def is_small_talk(text: str) -> bool:
    normalized = normalize_text(text).lower()
    normalized = normalized.replace("ı", "i").replace("ş", "s").replace("ğ", "g")
    normalized = normalized.replace("ü", "u").replace("ö", "o").replace("ç", "c")
    normalized = normalized.translate(str.maketrans("", "", ".,!?;:"))
    normalized = " ".join(normalized.split())

    greetings = {
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "merhaba", "selam", "gunaydin"
    }
    small_talk = {"how are you", "nasilsin", "naber"}
    thanks = {"thanks", "thank you", "tesekkurler"}

    return (
        normalized in greetings
        or normalized in small_talk
        or normalized in thanks
        or normalized in {"hello how are you", "hi how are you", "hey how are you", "merhaba nasilsin", "selam nasilsin"}
    )


def small_talk_reply(text: str) -> str:
    normalized = normalize_text(text).lower()
    normalized = normalized.replace("ı", "i").replace("ş", "s").replace("ğ", "g")
    normalized = normalized.replace("ü", "u").replace("ö", "o").replace("ç", "c")
    normalized = normalized.translate(str.maketrans("", "", ".,!?;:"))
    normalized = " ".join(normalized.split())

    turkish = any(word in normalized.split() for word in
                  ["merhaba", "selam", "gunaydin", "nasilsin", "naber"])

    if turkish:
        if "tesekkurler" in normalized:
            return "Rica ederim! Cihazınızla ilgili başka nasıl yardımcı olabilirim?"
        if "nasilsin" in normalized or "naber" in normalized:
            return "İyiyim, teşekkür ederim! Cihazınızda yaşadığınız sorunu anlatabilirsiniz."
        return "Merhaba! Lütfen cihazınızla ilgili sorunu anlatın, size yardımcı olayım."

    if normalized in {"thanks", "thank you"}:
        return "You're welcome! How else can I help with your appliance?"
    if "how are you" in normalized:
        return "I'm doing well, thanks! Tell me what is happening with your appliance."
    return "Hello! Please describe the issue with your appliance and I'll find the right support team."


def correct_product_name(token: str, known_names: list[str], cutoff: float = 0.78) -> str:
    """Minimum-edit-distance-style fuzzy correction for product names."""
    matches = get_close_matches(token.lower(), [x.lower() for x in known_names], n=1, cutoff=cutoff)
    return matches[0] if matches else token


def route_team(intent: str) -> str:
    value = intent.lower()
    if any(k in value for k in ("payment", "refund", "invoice", "cash", "credit", "account", "subscription")):
        return "Billing & Accounts"
    if any(k in value for k in ("delivery", "shipping", "track", "order", "return", "exchange")):
        return "Orders & Logistics"
    if any(k in value for k in ("calendar", "alarm", "datetime", "reminder", "timer")):
        return "Scheduling & Reminders"
    if any(k in value for k in ("music", "audio", "podcast", "radio", "volume", "play_")):
        return "Media Services"
    if any(k in value for k in ("iot", "smart", "cleaning", "coffee", "hue")):
        return "Smart Device Support"
    if any(k in value for k in ("cancel", "change", "update", "schedule", "appointment")):
        return "Customer Care"
    return "General Support"
