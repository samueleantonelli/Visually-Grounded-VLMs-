import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def extract_features(df, images_dir, model, processor, target_layers, text_only=False):
    # Extract last-token hidden states across target layers
    layer_embeddings = {l: [] for l in target_layers}

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Extracting (text_only={text_only})",
    ):
        description = row["description"]

        if text_only:
            messages = [{"role": "user", "content": [{"type": "text", "text": description}]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(text=[text_input], return_tensors="pt").to("cuda")
        else:
            image_path = f"{images_dir}/{row['image_file']}"
            raw_image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "text": description},
                    ],
                }
            ]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(
                text=[text_input],
                images=[raw_image],
                return_tensors="pt",
            ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Hidden states: tuple of shape (batch, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states

            for l in target_layers:
                # Last token bottleneck representation
                vec = hidden_states[l][0, -1, :].cpu().to(torch.float32).numpy()
                layer_embeddings[l].append(vec)

    for l in target_layers:
        layer_embeddings[l] = np.array(layer_embeddings[l])

    return layer_embeddings


def run_probing(X_train, y_train, X_val, y_val, X_test, y_test):
    # Linear probe with class balancing
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)

    preds_test = clf.predict(X_test)
    probs_test = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds_test)
    f1 = f1_score(y_test, preds_test)
    auroc = roc_auc_score(y_test, probs_test)
    return acc, f1, auroc, preds_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()

    # Load splits (must be partitioned strictly by image_id to prevent leakage)
    df = pd.read_csv(args.data_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    y_train = train_df["discriminative_label"].values
    y_val = val_df["discriminative_label"].values
    y_test = test_df["discriminative_label"].values

    # Model layers: even layers up to 26, plus final layer 27
    target_layers = list(range(0, 28, 2)) + [27]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 1. Multimodal probing extraction
    feats_train_mm = extract_features(
        train_df, args.images_dir, model, processor, target_layers, text_only=False
    )
    feats_val_mm = extract_features(
        val_df, args.images_dir, model, processor, target_layers, text_only=False
    )
    feats_test_mm = extract_features(
        test_df, args.images_dir, model, processor, target_layers, text_only=False
    )

    # 2. Layer 18 Probing (Multimodal, Random control, Text-only ablation)
    print("\n--- Layer 18 Probing Results ---")
    acc_mm, f1_mm, auroc_mm, _ = run_probing(
        feats_train_mm[18],
        y_train,
        feats_val_mm[18],
        y_val,
        feats_test_mm[18],
        y_test,
    )
    print(f"Multimodal (Img+Text)  -> Acc: {acc_mm:.2f}, F1: {f1_mm:.2f}, AUROC: {auroc_mm:.2f}")

    # Random labels baseline
    y_train_rand = np.random.permutation(y_train)
    acc_rnd, f1_rnd, auroc_rnd, _ = run_probing(
        feats_train_mm[18],
        y_train_rand,
        feats_val_mm[18],
        y_val,
        feats_test_mm[18],
        y_test,
    )
    print(f"Random Labels Control  -> Acc: {acc_rnd:.2f}, F1: {f1_rnd:.2f}, AUROC: {auroc_rnd:.2f}")

    # Text-only ablation
    feats_train_to = extract_features(
        train_df, args.images_dir, model, processor, [18], text_only=True
    )
    feats_val_to = extract_features(
        val_df, args.images_dir, model, processor, [18], text_only=True
    )
    feats_test_to = extract_features(
        test_df, args.images_dir, model, processor, [18], text_only=True
    )

    acc_to, f1_to, auroc_to, _ = run_probing(
        feats_train_to[18],
        y_train,
        feats_val_to[18],
        y_val,
        feats_test_to[18],
        y_test,
    )
    print(f"Text-Only Ablation     -> Acc: {acc_to:.2f}, F1: {f1_to:.2f}, AUROC: {auroc_to:.2f}")


if __name__ == "__main__":
    main()
