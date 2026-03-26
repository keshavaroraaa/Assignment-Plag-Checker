#!/usr/bin/env python3
"""
Assignment Plagiarism Risk Predictor
Uses TF-IDF + Cosine Similarity + Heuristic AI scoring to predict plagiarism risk.
Usage: python plag_risk_predictor.py <filename>
       python plag_risk_predictor.py <file1> <file2>   (compare two files)
"""

import sys
import os
import re
import math
import json
import hashlib
import datetime
from collections import Counter


# ─────────────────────────── Text Processing ────────────────────────────────

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

def read_file(path: str) -> str:
    """Read text from a file, supporting .txt and basic .py/.md/.html"""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
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


# ─────────────────────────── TF-IDF Engine ──────────────────────────────────

def tf(tokens: list[str]) -> dict:
    count = Counter(tokens)
    total = len(tokens) or 1
    return {w: c/total for w, c in count.items()}

def build_tfidf(tokens: list[str], corpus_tokens: list[str]) -> dict:
    tf_scores = tf(tokens)
    corpus_freq = Counter(corpus_tokens)
    total_corpus = len(corpus_tokens) or 1
    tfidf = {}
    for word, score in tf_scores.items():
        idf = math.log((total_corpus + 1) / (corpus_freq.get(word, 0) + 1)) + 1
        tfidf[word] = score * idf
    return tfidf

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────── AI Heuristic Features ──────────────────────────

def avg_sentence_length(text: str) -> float:
    sentences = get_sentences(text)
    if not sentences:
        return 0
    return sum(len(s.split()) for s in sentences) / len(sentences)

def vocabulary_richness(tokens: list[str]) -> float:
    """Type-Token Ratio (TTR) — low = possible copy-paste"""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def repeated_phrase_density(tokens: list[str]) -> float:
    """Ratio of repeated 4-grams — high = suspicious repetition"""
    ngrams = get_ngrams(tokens, 4)
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(ngrams)

def structural_uniformity(text: str) -> float:
    """Std-dev of sentence lengths — very low = template-like structure"""
    sentences = get_sentences(text)
    if len(sentences) < 3:
        return 0.5
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    # Normalize: low std → high uniformity score
    normalized = max(0.0, 1.0 - (std / (mean + 1)))
    return normalized

def passive_voice_ratio(text: str) -> float:
    """Passive voice usage — high in AI-generated or copied academic text"""
    passive_patterns = [
        r'\b(is|are|was|were|be|been|being)\s+\w+ed\b',
        r'\b(is|are|was|were)\s+\w+en\b',
    ]
    sentences = get_sentences(text)
    if not sentences:
        return 0.0
    passive_count = sum(
        1 for s in sentences
        if any(re.search(p, s.lower()) for p in passive_patterns)
    )
    return passive_count / len(sentences)

def detect_sudden_style_shift(text: str) -> float:
    """Detects abrupt changes in vocabulary complexity — sign of copy-paste"""
    sentences = get_sentences(text)
    if len(sentences) < 4:
        return 0.0
    complexities = [sum(len(w) for w in s.split()) / (len(s.split()) or 1) for s in sentences]
    shifts = 0
    for i in range(1, len(complexities)):
        if abs(complexities[i] - complexities[i-1]) > 3.0:
            shifts += 1
    return shifts / len(complexities)

def estimate_formality(tokens: list[str]) -> float:
    """Very high formality with limited personal pronouns → academic copy"""
    personal = {"i","me","my","mine","myself","we","us","our","ours"}
    ratio = sum(1 for t in tokens if t in personal) / (len(tokens) or 1)
    # Low personal pronoun ratio in long texts = formal/copied
    formality = 1.0 - min(ratio * 50, 1.0)
    return formality


# ─────────────────────────── Risk Scoring Engine ────────────────────────────

def score_single_file(text: str, tokens: list[str]) -> dict:
    """
    Multi-signal AI risk scoring for a single file.
    Returns a dict of feature scores and overall risk.
    """
    ttr = vocabulary_richness(tokens)
    rpd = repeated_phrase_density(tokens)
    unif = structural_uniformity(text)
    passive = passive_voice_ratio(text)
    shift = detect_sudden_style_shift(text)
    formality = estimate_formality(tokens)
    avg_sl = avg_sentence_length(text)

    # ── Normalize avg sentence length risk ──────────────────────────────────
    # Academic plagiarism often has very long sentences (>30 words avg)
    asl_risk = min((avg_sl / 30.0), 1.0) if avg_sl > 15 else 0.2

    # ── Weighted risk formula ────────────────────────────────────────────────
    #   Low TTR → risky      |  weight: 0.20
    #   High RPD → risky     |  weight: 0.20
    #   High uniformity → risky | weight: 0.15
    #   High passive → risky  | weight: 0.10
    #   High shift → risky    | weight: 0.15
    #   High formality → risky| weight: 0.10
    #   High asl_risk → risky | weight: 0.10

    risk = (
        (1 - ttr)    * 0.20 +
        rpd          * 0.20 +
        unif         * 0.15 +
        passive      * 0.10 +
        shift        * 0.15 +
        formality    * 0.10 +
        asl_risk     * 0.10
    )

    risk = round(min(max(risk, 0.0), 1.0), 4)

    return {
        "vocabulary_richness":    round(ttr, 4),
        "repeated_phrase_density": round(rpd, 4),
        "structural_uniformity":  round(unif, 4),
        "passive_voice_ratio":    round(passive, 4),
        "style_shift_score":      round(shift, 4),
        "formality_score":        round(formality, 4),
        "avg_sentence_length":    round(avg_sl, 2),
        "overall_risk_score":     risk,
    }

def score_two_files(text_a: str, text_b: str, tokens_a: list[str], tokens_b: list[str]) -> dict:
    """Compare two files using TF-IDF cosine similarity + ngram overlap."""
    combined = tokens_a + tokens_b

    vec_a = build_tfidf(tokens_a, combined)
    vec_b = build_tfidf(tokens_b, combined)
    cosine = cosine_similarity(vec_a, vec_b)

    # Bigram Jaccard similarity
    bg_a = set(get_ngrams(tokens_a, 2))
    bg_b = set(get_ngrams(tokens_b, 2))
    jaccard_2 = len(bg_a & bg_b) / (len(bg_a | bg_b) + 1e-9)

    # Trigram Jaccard
    tg_a = set(get_ngrams(tokens_a, 3))
    tg_b = set(get_ngrams(tokens_b, 3))
    jaccard_3 = len(tg_a & tg_b) / (len(tg_a | tg_b) + 1e-9)

    # Sentence-level exact match ratio
    sents_a = set(clean_text(s) for s in get_sentences(text_a))
    sents_b = set(clean_text(s) for s in get_sentences(text_b))
    exact_ratio = len(sents_a & sents_b) / (min(len(sents_a), len(sents_b)) + 1e-9)
    exact_ratio = min(exact_ratio, 1.0)

    # Final similarity score
    similarity = (
        cosine      * 0.35 +
        jaccard_2   * 0.25 +
        jaccard_3   * 0.25 +
        exact_ratio * 0.15
    )
    similarity = round(min(max(similarity, 0.0), 1.0), 4)

    return {
        "cosine_similarity":      round(cosine, 4),
        "bigram_jaccard":         round(jaccard_2, 4),
        "trigram_jaccard":        round(jaccard_3, 4),
        "exact_sentence_match":   round(exact_ratio, 4),
        "combined_similarity":    similarity,
    }


# ─────────────────────────── Risk Labelling ─────────────────────────────────

def risk_label(score: float) -> tuple[str, str]:
    if score < 0.20:
        return "✅  LOW RISK",        "Original work — minimal plagiarism indicators."
    elif score < 0.40:
        return "🟡  MODERATE RISK",   "Some indicators present; review recommended."
    elif score < 0.60:
        return "🟠  HIGH RISK",       "Multiple strong indicators — likely plagiarised."
    else:
        return "🔴  VERY HIGH RISK",  "Extremely high plagiarism probability. Investigate."


# ─────────────────────────── Report Generation ──────────────────────────────

SEPARATOR = "═" * 65

def bar(value: float, width: int = 40) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled) + f"  {value*100:.1f}%"

def print_single_report(path: str, text: str, tokens: list[str], scores: dict):
    label, desc = risk_label(scores["overall_risk_score"])
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{SEPARATOR}")
    print(f"  ASSIGNMENT PLAGIARISM RISK PREDICTOR")
    print(f"  Analyzed : {os.path.basename(path)}")
    print(f"  Words    : {len(tokens)}   |   Sentences: {len(get_sentences(text))}")
    print(f"  Time     : {ts}")
    print(SEPARATOR)

    print(f"\n  OVERALL RISK   {label}")
    print(f"  {desc}")
    print(f"\n  Risk Score  {bar(scores['overall_risk_score'])}")

    print(f"\n{SEPARATOR}")
    print("  FEATURE BREAKDOWN")
    print(SEPARATOR)

    features = [
        ("Vocabulary Richness",     scores["vocabulary_richness"],     True,  "Low richness = possible copy"),
        ("Repeated Phrase Density", scores["repeated_phrase_density"],  False, "High density = suspicious"),
        ("Structural Uniformity",   scores["structural_uniformity"],    False, "High = template-like"),
        ("Passive Voice Ratio",     scores["passive_voice_ratio"],      False, "High = formal/copied"),
        ("Style Shift Score",       scores["style_shift_score"],        False, "High = pasted sections"),
        ("Formality Score",         scores["formality_score"],          False, "High = no personal voice"),
    ]

    for name, val, is_good_high, hint in features:
        direction = "↑ good" if is_good_high else "↑ risky"
        print(f"\n  {name:<28}  {direction}")
        print(f"  {bar(val)}")
        print(f"  ↳ {hint}")

    print(f"\n  Avg. Sentence Length  :  {scores['avg_sentence_length']} words")
    print(f"\n{SEPARATOR}\n")

def print_comparison_report(path_a: str, path_b: str, single_a: dict, single_b: dict, comp: dict):
    sim = comp["combined_similarity"]
    label, desc = risk_label(sim)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{SEPARATOR}")
    print(f"  ASSIGNMENT PLAGIARISM COMPARATOR")
    print(f"  File A : {os.path.basename(path_a)}")
    print(f"  File B : {os.path.basename(path_b)}")
    print(f"  Time   : {ts}")
    print(SEPARATOR)

    print(f"\n  SIMILARITY VERDICT   {label}")
    print(f"  {desc}")
    print(f"\n  Combined Similarity  {bar(sim)}")

    print(f"\n{SEPARATOR}")
    print("  SIMILARITY SIGNALS")
    print(SEPARATOR)

    signals = [
        ("TF-IDF Cosine Similarity",  comp["cosine_similarity"]),
        ("Bigram Jaccard Overlap",    comp["bigram_jaccard"]),
        ("Trigram Jaccard Overlap",   comp["trigram_jaccard"]),
        ("Exact Sentence Match",      comp["exact_sentence_match"]),
    ]
    for name, val in signals:
        print(f"\n  {name:<30}  {bar(val)}")

    print(f"\n{SEPARATOR}")
    print("  INDIVIDUAL FILE RISKS")
    print(SEPARATOR)
    label_a, _ = risk_label(single_a["overall_risk_score"])
    label_b, _ = risk_label(single_b["overall_risk_score"])
    print(f"\n  {os.path.basename(path_a):<35} {label_a}  ({single_a['overall_risk_score']*100:.1f}%)")
    print(f"  {os.path.basename(path_b):<35} {label_b}  ({single_b['overall_risk_score']*100:.1f}%)")

    print(f"\n{SEPARATOR}\n")

def save_json_report(report: dict, base_path: str):
    out = base_path.replace(".txt","").replace(".py","").replace(".md","") + "_plag_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [✓] JSON report saved → {out}\n")


# ─────────────────────────── Entry Point ────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args:
        print("\n  Usage:")
        print("    python plag_risk_predictor.py <file.txt>")
        print("    python plag_risk_predictor.py <file1.txt> <file2.txt>")
        print()
        sys.exit(0)

    if len(args) == 1:
        path = args[0]
        text = read_file(path)
        tokens = tokenize(text)
        if len(tokens) < 30:
            print(f"[WARN] File has very few tokens ({len(tokens)}). Results may be unreliable.")
        scores = score_single_file(text, tokens)
        print_single_report(path, text, tokens, scores)
        report = {"file": path, "scores": scores, "timestamp": str(datetime.datetime.now())}
        save_json_report(report, path)

    elif len(args) == 2:
        path_a, path_b = args
        text_a, text_b = read_file(path_a), read_file(path_b)
        tok_a, tok_b = tokenize(text_a), tokenize(text_b)
        single_a = score_single_file(text_a, tok_a)
        single_b = score_single_file(text_b, tok_b)
        comp = score_two_files(text_a, text_b, tok_a, tok_b)
        print_comparison_report(path_a, path_b, single_a, single_b, comp)
        report = {
            "file_a": path_a, "file_b": path_b,
            "individual_a": single_a, "individual_b": single_b,
            "comparison": comp,
            "timestamp": str(datetime.datetime.now())
        }
        save_json_report(report, path_a)

    else:
        print("[ERROR] Provide 1 or 2 file paths only.")
        sys.exit(1)

if __name__ == "__main__":
    main()