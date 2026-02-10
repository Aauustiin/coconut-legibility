import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from monitor predictions"
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default="monitor_predictions.json",
        help="Path to the predictions JSON file (default: monitor_predictions.json)"
    )
    parser.add_argument(
        "--gsm8k-results",
        type=str,
        default="gsm8k_results.json",
        help="Path to the gsm8k results JSON file (default: gsm8k_results.json)"
    )
    parser.add_argument(
        "--latent-only",
        action="store_true",
        help="Only calculate metrics on questions with latent-only reasoning (no token-based thoughts)"
    )
    return parser.parse_args()


def get_latent_only_indices(gsm8k_results):
    """
    Identify question indices that contain only latent thoughts.

    A question has only latent thoughts if the model_reasoning ends with
    <|end-latent|> (possibly with whitespace) and has no additional tokens.
    """
    latent_only_indices = set()

    for result in gsm8k_results:
        reasoning = result["model_reasoning"]
        question_idx = result["question_idx"]

        # Check if reasoning ends with <|end-latent|> (possibly with trailing whitespace)
        if "<|end-latent|>" in reasoning:
            # Get everything after <|end-latent|>
            after_latent = reasoning.split("<|end-latent|>", 1)[1]
            # If only whitespace remains, this is a latent-only question
            if after_latent.strip() == "":
                latent_only_indices.add(question_idx)

    return latent_only_indices


def calculate_metrics(predictions):
    """
    Calculate accuracy, precision, recall, and F1 score.

    For this task:
    - True Positive (TP): Predicted YES and actually correct
    - True Negative (TN): Predicted NO and actually incorrect
    - False Positive (FP): Predicted YES but actually incorrect
    - False Negative (FN): Predicted NO but actually correct
    - UNKNOWN: Prediction could not be parsed
    """
    tp = 0  # True Positives
    tn = 0  # True Negatives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    unknown = 0  # Unknown predictions

    for pred in predictions:
        predicted = pred["prediction"]
        actual = pred["actual_correct"]

        if predicted == "UNKNOWN":
            unknown += 1
        elif predicted == "YES" and actual:
            tp += 1
        elif predicted == "NO" and not actual:
            tn += 1
        elif predicted == "YES" and not actual:
            fp += 1
        elif predicted == "NO" and actual:
            fn += 1

    # Calculate metrics (excluding UNKNOWN predictions)
    total_valid = tp + tn + fp + fn
    total_all = total_valid + unknown

    accuracy = (tp + tn) / total_all if total_all > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Balanced accuracy: average of recall on each class
    # Sensitivity (True Positive Rate) = TP / (TP + FN)
    # Specificity (True Negative Rate) = TN / (TN + FP)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        },
        "unknown_predictions": unknown,
        "total_valid_predictions": total_valid,
        "total_predictions": total_all
    }


def main():
    args = parse_args()

    print(f"Loading predictions from: {args.predictions_file}")
    with open(args.predictions_file, "r") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # Filter for latent-only questions if requested
    if args.latent_only:
        print(f"\nLoading gsm8k results from: {args.gsm8k_results}")
        with open(args.gsm8k_results, "r") as f:
            gsm8k_results = json.load(f)

        latent_only_indices = get_latent_only_indices(gsm8k_results)
        print(f"Found {len(latent_only_indices)} questions with latent-only reasoning")

        # Filter predictions to only include latent-only questions
        original_count = len(predictions)
        predictions = [p for p in predictions if p["question_idx"] in latent_only_indices]
        filtered_count = original_count - len(predictions)

        print(f"Filtered out {filtered_count} questions with mixed reasoning")
        print(f"Calculating metrics on {len(predictions)} latent-only predictions")

    print()

    # Calculate metrics
    metrics = calculate_metrics(predictions)

    # Add metadata about filtering
    metrics["latent_only_filter"] = args.latent_only
    if args.latent_only:
        metrics["filtered_count"] = filtered_count
        metrics["latent_only_count"] = len(predictions)

    # Print results
    print("=" * 80)
    if args.latent_only:
        print("METRICS (LATENT-ONLY QUESTIONS)")
    else:
        print("METRICS")
    print("=" * 80)
    print(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"Precision:         {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:            {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:          {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives:  {metrics['confusion_matrix']['true_positives']}")
    print(f"  True Negatives:  {metrics['confusion_matrix']['true_negatives']}")
    print(f"  False Positives: {metrics['confusion_matrix']['false_positives']}")
    print(f"  False Negatives: {metrics['confusion_matrix']['false_negatives']}")
    print()
    print(f"Unknown Predictions: {metrics['unknown_predictions']}")
    print(f"Valid Predictions:   {metrics['total_valid_predictions']}")
    print(f"Total Predictions:   {metrics['total_predictions']}")
    if args.latent_only:
        print()
        print(f"Latent-Only Questions: {len(predictions)}")
        print(f"Filtered Out (Mixed):  {filtered_count}")
    print("=" * 80)

    # Save metrics to file
    output_file = args.predictions_file.replace(".json", "_metrics.json")
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_file}")

    return metrics


if __name__ == "__main__":
    main()
