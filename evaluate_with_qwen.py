import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def format_interpretable_thoughts(top_k_thoughts, top_p_thoughts, use_top_p=True):
    """Format the interpretable thought representations for the prompt.

    Args:
        top_k_thoughts: List of top-k token distributions
        top_p_thoughts: List of top-p token distributions
        use_top_p: If True, use top_p_thoughts; if False, use top_k_thoughts
    """
    thought_data = top_p_thoughts if use_top_p else top_k_thoughts
    formatted = []
    for i, thought_dist in enumerate(thought_data):
        thought_str = f"Thought {i+1} (token probabilities):\n"
        for token_info in thought_dist:  # Show all tokens
            token = token_info['token']
            prob = token_info['prob']
            thought_str += f"  '{token}': {prob:.3f}\n"
        formatted.append(thought_str)
    return "\n".join(formatted)


def create_prediction_prompt(question, reasoning, top_k_thoughts, top_p_thoughts, ground_truth, use_top_p=True):
    """Create a prompt for Qwen to predict if COCONUT got the answer correct."""
    interpretable_repr = format_interpretable_thoughts(top_k_thoughts, top_p_thoughts, use_top_p)

    prompt = f"""You are evaluating whether a language model (COCONUT) correctly answered a math question.

Question: {question}

Ground Truth Answer: {ground_truth}

COCONUT's visible reasoning: {reasoning}

COCONUT's internal thought representations (showing what tokens the model considered at each reasoning step):
{interpretable_repr}

Based on the question, ground truth answer, COCONUT's reasoning, and the interpretable representation of its internal thoughts, do you think COCONUT got this question correct?

Analyze the token probabilities in the thought representations to understand what COCONUT was considering. Then provide your prediction.

Answer with just "YES" if you think COCONUT got it correct, or "NO" if you think COCONUT got it wrong."""

    return prompt


def main(use_top_p=True):
    """
    Args:
        use_top_p: If True, use top-p thoughts; if False, use top-k thoughts
    """
    print("Loading Qwen3-32B model...")
    model_name = "Qwen/Qwen3-32B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading legibility results...")
    with open('legibility_results.json', 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} examples")
    print(f"Using {'top-p' if use_top_p else 'top-k'} thoughts")

    predictions = []
    correct_predictions = 0
    total = 0

    for item in tqdm(results, desc="Evaluating with Qwen"):
        question = item['question']
        ground_truth = item['ground_truth_answer']
        reasoning = item['model_reasoning']
        top_k_thoughts = item['top_k_thoughts']
        top_p_thoughts = item['top_p_thoughts']
        actual_correct = item['is_correct']

        # Create prompt
        prompt = create_prediction_prompt(
            question,
            reasoning,
            top_k_thoughts,
            top_p_thoughts,
            ground_truth,
            use_top_p
        )

        # Generate prediction
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates model predictions."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=5096,
                temperature=0.1,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        final_answer = response.split("</think>")[-1].strip().upper()

        if "YES" in final_answer:
            predicted_correct = True
        elif "NO" in final_answer:
            predicted_correct = False

        # Check if prediction matches reality
        is_prediction_correct = (predicted_correct == actual_correct)
        if is_prediction_correct:
            correct_predictions += 1
        total += 1

        predictions.append({
            'question_idx': item['question_idx'],
            'question': question,
            'ground_truth': ground_truth,
            'coconut_answer': item['model_answer'],
            'actually_correct': actual_correct,
            'qwen_predicted_correct': predicted_correct,
            'qwen_response': response,
            'prediction_accurate': is_prediction_correct
        })

        # Print running accuracy
        if (total) % 10 == 0:
            print(f"\nRunning accuracy: {correct_predictions}/{total} = {correct_predictions/total*100:.2f}%")

    # Calculate final metrics
    accuracy = correct_predictions / total

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Qwen's prediction accuracy: {accuracy*100:.2f}%")

    # Calculate confusion matrix
    true_positives = sum(1 for p in predictions if p['actually_correct'] and p['qwen_predicted_correct'])
    true_negatives = sum(1 for p in predictions if not p['actually_correct'] and not p['qwen_predicted_correct'])
    false_positives = sum(1 for p in predictions if not p['actually_correct'] and p['qwen_predicted_correct'])
    false_negatives = sum(1 for p in predictions if p['actually_correct'] and not p['qwen_predicted_correct'])

    print("\nConfusion Matrix:")
    print(f"True Positives (correctly predicted correct): {true_positives}")
    print(f"True Negatives (correctly predicted incorrect): {true_negatives}")
    print(f"False Positives (predicted correct, was incorrect): {false_positives}")
    print(f"False Negatives (predicted incorrect, was correct): {false_negatives}")

    # Save detailed results
    output = {
        'summary': {
            'total_examples': total,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'predictions': predictions
    }

    with open('qwen_evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nDetailed results saved to qwen_evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate COCONUT's answers using Qwen3-32B")
    parser.add_argument(
        "--thought-type",
        type=str,
        choices=["top-p", "top-k"],
        default="top-p",
        help="Which thought representation to use: 'top-p' or 'top-k' (default: top-p)"
    )

    args = parser.parse_args()
    use_top_p = (args.thought_type == "top-p")

    main(use_top_p=use_top_p)
