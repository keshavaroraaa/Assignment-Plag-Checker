#!/usr/bin/env python3
import sys
import os
import re
import math
import json
import datetime
from collections import Counter

# ─────────────────────────── Constants & Setup ──────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","that",
    "this","these","those","it","its","we","i","you","he","she","they","my",
    "our","your","his","her","their","as","if","then","than","so","yet","both",
    "either","not","no","nor","by","from","into","through","about","after",
    "before","between","during","while","because","although","since","unless",
    "until","when","where","which","who","whom","whose","how","what","there",
    "here","just","also","more","some","all","each","every","any","such","only",
}

# ─────────────────────────── Core Logic Functions ───────────────────────────

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    return [w for w in clean_text(text).split() if w not in STOPWORDS and len(w) > 2]

def get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def get_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]

# ─────────────────────────── Metrics ────────────────────────────────────────

def vocabulary_richness(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def repeated_phrase_density(tokens: list[str]) -> float:
    ngrams = get_ngrams(tokens, 4)
    if not ngrams: return 0.0
    counts = Counter(ngrams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(ngrams)

def structural_uniformity(text: str) -> float:
    sentences = get_sentences(text)
    if len(sentences) < 3: return 0.5
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    return max(0.0, 1.0 - (std / (mean + 1)))

def passive_voice_ratio(text: str) -> float:
    passive_patterns = [r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', r'\b(is|are|was|were)\s+\w+en\b']
    sentences = get_sentences(text)
    if not sentences: return 0.0
    passive_count = sum(1 for s in sentences if any(re.search(p, s.lower()) for p in passive_patterns))
    return passive_count / len(sentences)

def detect_sudden_style_shift(text: str) -> float:
    sentences = get_sentences(text)
    if len(sentences) < 4: return 0.0
    complexities = [sum(len(w) for w in s.split()) / (len(s.split()) or 1) for s in sentences]
    shifts = sum(1 for i in range(1, len(complexities)) if abs(complexities[i] - complexities[i-1]) > 3.0)
    return shifts / len(complexities)

def estimate_formality(tokens: list[str]) -> float:
    personal = {"i","me","my","mine","myself","we","us","our","ours"}
    ratio = sum(1 for t in tokens if t in personal) / (len(tokens) or 1)
    return 1.0 - min(ratio * 50, 1.0)

# ─────────────────────────── Scoring Engine ─────────────────────────────────

def score_single_file(text: str, tokens: list[str]) -> dict:
    ttr = vocabulary_richness(tokens)
    rpd = repeated_phrase_density(tokens)
    unif = structural_uniformity(text)
    passive = passive_voice_ratio(text)
    shift = detect_sudden_style_shift(text)
    formality = estimate_formality(tokens)
    
    sentences = get_sentences(text)
    avg_sl = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    asl_risk = min((avg_sl / 30.0), 1.0) if avg_sl > 15 else 0.2

    risk = ((1 - ttr) * 0.20 + rpd * 0.20 + unif * 0.15 + passive * 0.10 + shift * 0.15 + formality * 0.10 + asl_risk * 0.10)
    
    return {
        "vocabulary_richness": round(ttr, 4),
        "repeated_phrase_density": round(rpd, 4),
        "structural_uniformity": round(unif, 4),
        "passive_voice_ratio": round(passive, 4),
        "style_shift_score": round(shift, 4),
        "formality_score": round(formality, 4),
        "overall_risk_score": round(min(max(risk, 0.0), 1.0), 4)
    }

# ─────────────────────────── CLI Entry Point ────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plag_predictor.py <filename>")
    else:
        path = sys.argv[1]
        raw_text = read_file(path)
        toks = tokenize(raw_text)
        results = score_single_file(raw_text, toks)
        print(json.dumps(results, indent=2))