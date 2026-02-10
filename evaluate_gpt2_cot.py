import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
from utils import get_unique_filepath


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 CoT model on GSM8k")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output files (default: outputs)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialise tokeniser with special tokens
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model weights
    saved_weights = torch.load(
        "/users/cns542/scratch/coconut/gsm-cot/checkpoint_7",
        map_location=device
    )
    model.load_state_dict(saved_weights, strict=False)

    # Move to GPU and set eval mode
    model = model.to(device)
    model.eval()

    # Load GSM8k test dataset
    gsm_data = json.load(open("data/gsm_test.json"))

    # Store results
    results = []

    # Process each question in the dataset
    for idx, sample in enumerate(gsm_data):
        question = sample["question"]
        ground_truth_answer = sample["answer"]

        print(f"\n{'='*80}")
        print(f"Question {idx+1}/{len(gsm_data)}")
        print(f"{'='*80}")
        print(f"Q: {question}")
        print(f"Ground truth answer: {ground_truth_answer}")
        print()

        input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        # Generate a response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128
            )

        # Extract answer and reasoning
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer_output = output_text.split("#")[-1].replace(",", "").strip()
        reasoning = output_text.split("#")[0].strip() if "#" in output_text else ""
        if question in reasoning:
            reasoning = reasoning.replace(question, "").strip()
        is_correct = answer_output == ground_truth_answer

        print("-" * 80)
        print(f"Model output:\n{output_text}")
        print()
        print(f"Extracted answer: {answer_output}")
        print(f"Correct: {is_correct}")
        print()

        results.append({
            "question_idx": idx,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "model_reasoning": reasoning,
            "model_answer": answer_output,
            "is_correct": is_correct,
        })

    # Save results with unique filename
    results_path = get_unique_filepath(args.output_dir, "gpt2_cot_gsm_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()