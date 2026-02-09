import json
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def parse_filename(filename):
    """
    Parse metrics filename to extract model and representation type.
    Example: 'Qwen3-14B_CoT_Predictions_metrics.json' -> ('Qwen3-14B', 'CoT')
    """
    basename = os.path.basename(filename)
    # Remove '_Predictions_metrics.json' suffix
    name = basename.replace('_Predictions_metrics.json', '')

    # Split by underscore to separate model and representation
    parts = name.split('_')

    # Handle cases like 'Qwen3-14B' and 'Top-K_5'
    if len(parts) >= 2:
        # Last part is representation type (e.g., 'CoT', 'Raw', '5' for Top-K)
        # Everything before is the model name

        # Special handling for 'Top-K_5' format
        if len(parts) >= 3 and parts[-2] == 'Top-K':
            model = '_'.join(parts[:-2])
            representation = f"{parts[-2]}_{parts[-1]}"
        else:
            model = '_'.join(parts[:-1])
            representation = parts[-1]
    else:
        model = parts[0]
        representation = "Unknown"

    return model, representation


def load_metrics():
    """Load all metrics files and organize by model and representation."""
    metrics_files = glob.glob('*_metrics.json')

    data = defaultdict(dict)

    for filepath in metrics_files:
        model, representation = parse_filename(filepath)

        with open(filepath, 'r') as f:
            metrics = json.load(f)

        data[model][representation] = {
            'accuracy': metrics.get('accuracy', 0),
            'balanced_accuracy': metrics.get('balanced_accuracy', 0),
            'f1_score': metrics.get('f1_score', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0)
        }

    return data


def plot_metrics(data):
    """Create line plots for accuracy and F1 score."""
    # Define representation types
    representation_order = ['Raw', 'CoT', 'Top-K_5']

    # Sort models by size (extract number from model name)
    models = sorted(data.keys(), key=lambda x: float(x.split('-')[-1].replace('B', '')))

    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot balanced accuracy - one line per representation type
    for rep in representation_order:
        balanced_accuracies = []
        model_labels = []

        for model in models:
            if rep in data[model]:
                balanced_accuracies.append(data[model][rep]['balanced_accuracy'] * 100)  # Convert to percentage
                model_labels.append(model)

        if balanced_accuracies:
            ax1.plot(model_labels, balanced_accuracies, marker='o', label=rep, linewidth=2, markersize=8)

    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Balanced Accuracy (%)', fontsize=12)
    ax1.set_title('Balanced Accuracy by Model and Representation', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Plot F1 score - one line per representation type
    for rep in representation_order:
        f1_scores = []
        model_labels = []

        for model in models:
            if rep in data[model]:
                f1_scores.append(data[model][rep]['f1_score'] * 100)  # Convert to percentage
                model_labels.append(model)

        if f1_scores:
            ax2.plot(model_labels, f1_scores, marker='o', label=rep, linewidth=2, markersize=8)

    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('F1 Score (%)', fontsize=12)
    ax2.set_title('F1 Score by Model and Representation', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()

    # Save the plot
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved to: metrics_comparison.png")


def print_summary(data):
    """Print a summary table of the metrics."""
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)

    for model in sorted(data.keys()):
        print(f"\n{model}:")
        print(f"  {'Representation':<15} {'Bal. Acc':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
        print("  " + "-" * 63)

        for rep in ['Raw', 'CoT', 'Top-K_5']:
            if rep in data[model]:
                m = data[model][rep]
                print(f"  {rep:<15} {m['balanced_accuracy']*100:>10.2f}%  {m['f1_score']*100:>10.2f}%  {m['precision']*100:>10.2f}%  {m['recall']*100:>10.2f}%")

    print("\n" + "=" * 80)


def main():
    print("Loading metrics files...")
    data = load_metrics()

    print(f"Found metrics for {len(data)} models")

    # Print summary table
    print_summary(data)

    # Create plots
    print("\nGenerating plots...")
    plot_metrics(data)


if __name__ == "__main__":
    main()
