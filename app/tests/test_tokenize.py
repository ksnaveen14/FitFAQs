# tests/test_tokenize.py

from retrieve import tokenize


def test_strips_punctuation():
    assert tokenize("How much protein per day?") == ["how", "much", "protein", "per", "day"]


def test_lowercases_and_handles_hyphens():
    assert tokenize("Zone-2 Cardio, 60-70%") == ["zone", "2", "cardio", "60", "70"]


def test_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("?!...") == []