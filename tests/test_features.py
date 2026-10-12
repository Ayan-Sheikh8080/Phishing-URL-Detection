from src.features import FEATURE_NAMES, extract_features


def test_feature_keys_match():
    assert list(extract_features("http://a.com").keys()) == FEATURE_NAMES


def test_https_flag():
    assert extract_features("https://example.com")["has_https"] == 1
    assert extract_features("http://example.com")["has_https"] == 0


def test_ip_host():
    assert extract_features("http://192.168.1.1/login")["has_ip_host"] == 1


def test_suspicious_words():
    f = extract_features("http://paypal-login-verify.xyz/account")
    assert f["suspicious_word_count"] >= 3


def test_malformed_url_does_not_crash():
    assert extract_features("http://[bad")["url_length"] > 0