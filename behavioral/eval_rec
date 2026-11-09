import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args():
    # CLI arguments
    parser = argparse.ArgumentParser(
        description="Run Referring Expression Comprehension evaluation."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to trial annotations JSON/CSV.",
    )
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Path to base images directory."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="rec_results.csv",
        help="Path to save per-trial results.",
    )
    return parser.parse_args()


def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def parse_bbox_from_response(text):
    # Regex search for bounding box formatted as [Xmin, Ymin, Xmax, Ymax]
    pattern = r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]"
    match = re.search(pattern, text)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return None


def main():
    args = parse_args()

    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Load dataset
    df = (
        pd.read_csv(args.data_path)
        if args.data_path.endswith(".csv")
        else pd.read_json(args.data_path)
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating REC"):
        image_path = f"{args.images_dir}/{row['image_file']}"
        raw_image = Image.open(image_path).convert("RGB")
        target_description = row["description"]
        gold_box = row["gold_box"]  # Expected [xmin, ymin, xmax, ymax]

        # Construct prompt matching the behavioral evaluation setup
        prompt = (
            f"Identify the target in the provided image given the description. "
            f"Output your answer in a bounding box in the format [Xmin, Ymin, Xmax, Ymax]\n"
            f'Description: "{target_description}"'
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            images=[raw_image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        pred_box = parse_bbox_from_response(response)
        iou = calculate_iou(pred_box, gold_box) if pred_box else 0.0

        results.append(
            {
                "image_id": row["image_id"],
                "speaker": row["speaker"],
                "condition": row["condition"],
                "distractor_count": row.get("distractor_count", 0),
                "gold_box": gold_box,
                "pred_box": pred_box,
                "raw_response": response,
                "iou": iou,
                "success_05": int(iou >= 0.50),
                "success_075": int(iou >= 0.75),
            }
        )

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_path, index=False)

    # Print summary metrics
    print("\n--- Summary Evaluation Metrics ---")
    print(f"Overall Accuracy (IoU >= 0.50): {res_df['success_05'].mean() * 100:.2f}%")
    print(
        f"Overall Accuracy (IoU >= 0.75): {res_df['success_075'].mean() * 100:.2f}%"
    )
    print(f"Mean IoU: {res_df['iou'].mean():.4f}")


if __name__ == "__main__":
    main()
