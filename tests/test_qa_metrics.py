"""Tests for QA metrics F1 and BLEU-1."""

import pytest

from src.eval.metrics import bleu1_score, f1_score


# 1. F1 metric
def test_f1_exact_match():
    assert f1_score("hello world", "hello world") == pytest.approx(1.0)
    assert f1_score("User prefers dark mode", "User prefers dark mode") == pytest.approx(1.0)
    # Case-insensitive exact
    assert f1_score("Hello World", "hello world") == pytest.approx(1.0)


def test_f1_partial_overlap():
    # pred has 2 tokens, ref has 3, overlap 1
    assert f1_score("hello world", "hello there") == pytest.approx(0.5)
    # pred "a b c", ref "a b d" => overlap 2, P=2/3, R=2/3 => F1=2/3
    assert f1_score("a b c", "a b d") == pytest.approx(2 / 3)


def test_f1_no_overlap():
    assert f1_score("hello", "world") == pytest.approx(0.0)
    assert f1_score("a b c", "x y z") == pytest.approx(0.0)


def test_f1_empty_prediction():
    assert f1_score("", "hello world") == pytest.approx(0.0)
    assert f1_score("", "") == pytest.approx(1.0)


def test_f1_empty_reference():
    assert f1_score("hello", "") == pytest.approx(0.0)
    assert f1_score("", "") == pytest.approx(1.0)


def test_f1_punctuation_and_case():
    # punctuation stripped, case-insensitive
    assert f1_score("Hello, world!", "hello world") == pytest.approx(1.0)
    assert f1_score("User prefers dark-mode.", "user prefers dark mode") == pytest.approx(1.0)


# BLEU-1
def test_bleu1_exact_match():
    assert bleu1_score("hello world", "hello world") == pytest.approx(1.0)
    assert bleu1_score("User lives in Berlin", "User lives in Berlin") == pytest.approx(1.0)


def test_bleu1_partial():
    # pred "hello world", ref "hello there" => matched 1/2, BP=1 (same len) => 0.5
    assert bleu1_score("hello world", "hello there") == pytest.approx(0.5)
    # longer pred than ref, no brevity penalty
    assert bleu1_score("a b c d", "a b") == pytest.approx(0.5)  # matched 2/4=0.5, BP=1


def test_bleu1_no_overlap():
    assert bleu1_score("hello", "world") == pytest.approx(0.0)


def test_bleu1_empty_cases():
    assert bleu1_score("", "") == pytest.approx(1.0)
    assert bleu1_score("", "hello") == pytest.approx(0.0)
    assert bleu1_score("hello", "") == pytest.approx(0.0)


def test_bleu1_brevity_penalty():
    # pred shorter than ref => BP <1
    # pred "a b", ref "a b c d" => p1=1.0 (2/2 matched), BP=exp(1-4/2)=exp(-1)=0.3679 => BLEU=0.3679
    import math

    expected = math.exp(1 - 4 / 2) * 1.0
    assert bleu1_score("a b", "a b c d") == pytest.approx(expected)
    # pred longer than ref => BP=1
    assert bleu1_score("a b c d", "a b") == pytest.approx(0.5)  # as above, BP=1


def test_bleu1_and_f1_deterministic():
    for fn in (f1_score, bleu1_score):
        a = fn("hello world", "hello there")
        b = fn("hello world", "hello there")
        assert a == b
