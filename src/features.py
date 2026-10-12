import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "secure", "account", "update",
    "bank", "paypal", "confirm", "password", "webscr", "wallet",
)
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def extract_features(url: str) -> dict:
    raw = str(url).strip()
    try:
        p = urlparse(raw if "://" in raw else "http://" + raw)
        host = p.hostname or ""
        path, query = p.path or "", p.query or ""
    except ValueError:  # malformed URL
        host, path, query = "", "", ""

    is_ip = bool(IP_RE.match(host))
    parts = host.split(".") if host else []
    n = max(len(raw), 1)
    digits = sum(c.isdigit() for c in raw)
    letters = sum(c.isalpha() for c in raw)
    lower = raw.lower()

    return {
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/"),
        "num_at": raw.count("@"),
        "num_question": raw.count("?"),
        "num_equals": raw.count("="),
        "num_ampersand": raw.count("&"),
        "num_percent": raw.count("%"),
        "num_digits": digits,
        "num_letters": letters,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "has_https": int(lower.startswith("https://")),
        "has_ip_host": int(is_ip),
        "subdomain_count": 0 if is_ip else max(len(parts) - 2, 0),
        "tld_length": 0 if is_ip or not parts else len(parts[-1]),
        "num_query_params": len([q for q in query.split("&") if q]),
        "suspicious_word_count": sum(w in lower for w in SUSPICIOUS_WORDS),
    }


FEATURE_NAMES = list(extract_features("http://example.com").keys())