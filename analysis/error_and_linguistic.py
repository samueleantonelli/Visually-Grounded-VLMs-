import argparse
import re
import numpy as np
import pandas as pd
from scipy.stats import chi2


def extract_linguistic_stats(df):
    spatial_keywords = {
        "left",
        "right",
        "top",
        "bottom",
        "front",
        "behind",
        "next",
        "middle",
        "center",
        "side",
        "corner",
    }
    ordinal_keywords = {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "1st",
        "2nd",
        "3rd",
        "last",
    }
    color_keywords = {
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "brown",
        "orange",
        "pink",
        "gray",
        "grey",
    }
    definite_articles = {"the"}

    stats = []
    for _, row in df.iterrows():
        text = str(row["description"]).lower()
        tokens = re.findall(r"\b\w+\b", text)
        length = len(tokens)

        n_spatial = sum(1 for t in tokens if t in spatial_keywords)
        n_ordinal = sum(1 for t in tokens if t in ordinal_keywords)
        n_color = sum(1 for t in tokens if t in color_keywords)
        n_def = sum(1 for t in tokens if t in definite_articles)

        stats.append(
            {
                "label": row["discriminative_label"],
                "spatial": n_spatial,
                "ordinal": n_ordinal,
                "definite": n_def,
                "color": n_color,
                "length": length,
            }
        )

    feat_df = pd.DataFrame(stats)
    grouped = feat_df.groupby("label").mean()
    print("=== Linguistic Feature Distribution (Average per Sentence) ===")
    print(grouped.T)


def run_mcnemar(df):
    # Cross-tabulate multimodal vs unimodal predictions
    # b: Multimodal correct, Text-only incorrect (Vision rescues)
    # c: Multimodal incorrect, Text-only correct (Vision interferes)
    b = len(df[(df["mm_correct"] == 1) & (df["to_correct"] == 0)])
    c = len(df[(df["mm_correct"] == 0) & (df["to_correct"] == 1)])
    agree_correct = len(df[(df["mm_correct"] == 1) & (df["to_correct"] == 1)])
    agree_wrong = len(df[(df["mm_correct"] == 0) & (df["to_correct"] == 0)])

    print("\n=== Contingency Matrix ===")
    print(f"Both Correct: {agree_correct}")
    print(f"Both Incorrect: {agree_wrong}")
    print(f"Vision Rescues Text (b): {b}")
    print(f"Vision Interferes (c): {c}")

    # McNemar chi-square with continuity correction
    chi2_val = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = chi2.sf(chi2_val, df=1)

    print(f"\nMcNemar Test: chi2 = {chi2_val:.3f}, p-value = {p_val:.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_results_csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.test_results_csv)
    extract_linguistic_stats(df)

    if "mm_correct" in df.columns and "to_correct" in df.columns:
        run_mcnemar(df)


if __name__ == "__main__":
    main()
