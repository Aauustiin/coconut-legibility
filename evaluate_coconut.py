import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from coconut import Coconut

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialise tokeniser with special tokens
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Initialise model
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # Load model weights
    saved_weights = torch.load(
        "/users/cns542/scratch/coconut/gsm-coconut-true/checkpoint_11",
        map_location=device
    )
    model.load_state_dict(saved_weights, strict=False)

    # Move to GPU and set eval mode
    model = model.to(device)
    model.eval()

    # Load GSM8k test dataset
    gsm_data = json.load(open("data/gsm_test.json"))

    # Number of latent tokens to use
    num_latent_tokens = 3

    # Store results
    results = []
    raw_thoughts = []
    logit_thoughts = []

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

        # Tokenize prompt with latent tokens
        prompt = f"{question}\n<|start-latent|>" + "<|latent|>" * num_latent_tokens + "<|end-latent|>"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        # Generate a response
        with torch.no_grad():
            output_ids, latent_hidden_states = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                output_latent_hidden_states=True
            )

        # Process thoughts and collect results
        raw_thought_sequence = []
        logit_thought_sequence = []

        lm_head = model.base_causallm.lm_head
        for state in latent_hidden_states:
            raw_thought_sequence.append(state.cpu())
            thought_logits = lm_head(state)
            logit_thought_sequence.append(thought_logits.cpu())

        raw_thoughts.append(raw_thought_sequence)
        logit_thoughts.append(logit_thought_sequence)

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

    # Save results
    with open("legibility_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to legibility_results.json")

    # Save raw thoughts and logit thoughts separately
    torch.save(raw_thoughts, "raw_thoughts.pt")
    print(f"Raw thoughts saved to raw_thoughts.pt")

    torch.save(logit_thoughts, "logit_thoughts.pt")
    print(f"Logit thoughts saved to logit_thoughts.pt")

if __name__ == "__main__":
    main()