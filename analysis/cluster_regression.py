import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec_results_csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.rec_results_csv)

    # Condition encoding: set 'Model_Default' as reference category
    # Ensure columns match: success_05, condition_type, distractor_count, image_id
    formula = (
        "success_05 ~ C(condition, Treatment(reference='Model Default')) + distractor_count"
    )

    print("=== Full Dataset Regression (Huber-White cluster-robust on image_id) ===")
    model_full = smf.logit(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["image_id"]}
    )
    print(model_full.summary())

    # Calculate Odds Ratios and 95% Confidence Intervals
    params = model_full.params
    conf = model_full.conf_int()
    conf["OR"] = params
    conf.columns = ["2.5%", "97.5%", "OR"]
    print("\nOdds Ratios:")
    print(np.exp(conf))

    print("\n=== Subset with >= 1 Distractor ===")
    subset_df = df[df["distractor_count"] >= 1].reset_index(drop=True)
    model_sub = smf.logit(formula, data=subset_df).fit(
        cov_type="cluster", cov_kwds={"groups": subset_df["image_id"]}
    )
    print(model_sub.summary())

    conf_sub = model_sub.conf_int()
    conf_sub["OR"] = model_sub.params
    conf_sub.columns = ["2.5%", "97.5%", "OR"]
    print("\nOdds Ratios (Competitor Subset):")
    print(np.exp(conf_sub))


if __name__ == "__main__":
    main()
