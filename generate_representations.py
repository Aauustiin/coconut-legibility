import torch
import argparse
import json
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Process thoughts with top-k, top-p, or raw sampling")
    parser.add_argument(
        "--thought-type",
        type=str,
        choices=["top-k", "top-p", "raw"],
        default="top-k",
        help="Which sampling method to use: 'top-k', 'top-p', or 'raw' (default: top-k)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Value of k for top-k sampling (default: 5)"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.9,
        help="Value of p for top-p sampling (default: 0.9)"
    )
    return parser.parse_args()


def topp(probs, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    num_tokens = (cumsum_probs <= p).sum() + 1
    selected_indices = sorted_indices[:num_tokens]
    selected_probs = sorted_probs[:num_tokens]
    return selected_probs, selected_indices


def format_representation(representation):
    """Convert representation to a readable string format."""
    formatted = []
    for thought_idx, thought in enumerate(representation):
        formatted.append(f"Thought {thought_idx + 1}:")
        for token_data in thought:
            token = token_data["token"]
            prob = token_data["prob"]
            formatted.append(f"  '{repr(token)}': {prob:.4f}")
    return "\n".join(formatted)

def main():
    # Initialise tokeniser with special tokens
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    args = parse_args()

    # Load raw thoughts or logit thoughts based on thought type
    if args.thought_type == "raw":
        raw_thoughts = torch.load("raw_thoughts.pt")
    else:
        logit_thoughts = torch.load("logit_thoughts.pt")

        prob_thoughts = []
        for sequence in logit_thoughts:
            sequence_probs = []
            for thought in sequence:
                probs = torch.softmax(thought, dim=-1)
                sequence_probs.append(probs.cpu())
            prob_thoughts.append(sequence_probs)

    representations = []
    text_representations = []

    if args.thought_type == "raw":
        filename = f"representations_{args.thought_type}.pt"
        text_filename = f"representations_{args.thought_type}.json"
        for sequence in raw_thoughts:
            representation = []
            for thought in sequence:
                thought_values = thought.flatten().tolist()
                rounded_values = [round(val, 4) for val in thought_values]
                representation.append(rounded_values)
            representations.append(representation)
            # Format text representation
            formatted = []
            for thought_idx, thought_values in enumerate(representation):
                formatted.append(f"Thought {thought_idx + 1}:")
                values_str = ", ".join([f"{val:.4f}" for val in thought_values])
                formatted.append(f"  [{values_str}]")
            text_representations.append("\n".join(formatted))
    elif args.thought_type == "top-k":
        filename = f"representations_{args.thought_type}_{args.k}.pt"
        text_filename = f"representations_{args.thought_type}_{args.k}.json"
        for sequence in prob_thoughts:
            representation = []
            for thought in sequence:
                top_k_probs, top_k_indices = torch.topk(thought, k=args.k)
                top_k_data = [
                    {"token": tokenizer.decode(top_k_indices[j].item()),
                        "prob": top_k_probs[j].item()}
                    for j in range(len(top_k_indices))
                ]
                representation.append(top_k_data)
            representations.append(representation)
            text_representations.append(format_representation(representation))
    elif args.thought_type == "top-p":
        filename = f"representations_{args.thought_type}_{args.p}.pt"
        text_filename = f"representations_{args.thought_type}_{args.p}.json"
        for sequence in prob_thoughts:
            representation = []
            for thought in sequence:
                top_k_probs, top_k_indices = topp(thought, p=args.p)
                top_k_data = [
                    {"token": tokenizer.decode(top_k_indices[j].item()),
                        "prob": top_k_probs[j].item()}
                    for j in range(len(top_k_indices))
                ]
                representation.append(top_k_data)
            representations.append(representation)
            text_representations.append(format_representation(representation))

    # Save original representations
    torch.save(representations, filename)
    print(f"Saved representations to {filename}")

    # Save text representations
    with open(text_filename, "w") as f:
        json.dump(text_representations, f, indent=2)
    print(f"Saved text representations to {text_filename}")


if __name__ == "__main__":
    main()  