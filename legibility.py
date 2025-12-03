import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from coconut import Coconut

def topp(probs, p=0.9):

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    
    num_tokens = (cumsum_probs <= p).sum() + 1
    
    selected_indices = sorted_indices[:num_tokens]
    selected_probs = sorted_probs[:num_tokens]
    
    return selected_probs, selected_indices

def main():
    # Setup GPUs
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

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
    saved_weights = torch.load("/users/cns542/scratch/coconut/gsm-coconut-true/checkpoint_11", map_location=f"cuda:{local_rank}")
    model.load_state_dict(saved_weights, strict=False)

    # Move to GPU and set eval mode
    model = model.to(local_rank)
    model.eval()

    # Load GSM8k test dataset
    gsm_data = json.load(open("data/gsm_test.json"))

    # Number of latent tokens to use (you can adjust this)
    num_latent_tokens = 3

    # Store results
    results = []

    # Process each question in the dataset (first 10 for quick testing)
    for idx, sample in enumerate(gsm_data[:10]):
        question = sample["question"]
        ground_truth_answer = sample["answer"]

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Question {idx+1}/{len(gsm_data[:10])}")
            print(f"{'='*80}")
            print(f"Q: {question}")
            print(f"Ground truth answer: {ground_truth_answer}")
            print()

        # Tokenize prompt with latent tokens
        prompt = f"{question}\n<|start-latent|>" + "<|latent|>" * num_latent_tokens + "<|end-latent|>"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(local_rank)
        attention_mask = torch.ones_like(input_ids)

        # Generate a response
        with torch.no_grad():
            output_ids, latent_hidden_states = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                output_latent_hidden_states=True
            )

        lm_head = model.base_causallm.lm_head

        # Process thoughts and collect results
        top_k_thoughts = []
        top_p_thoughts = []
        raw_thoughts = []

        for i, hidden in enumerate(latent_hidden_states):
            thought_logits = lm_head(hidden)
            thought_probs = torch.softmax(thought_logits, dim=-1)

            # Raw continuous thought (hidden state as list)
            raw_thoughts.append(hidden.cpu().tolist())

            # Top-k
            top_k_probs, top_k_indices = torch.topk(thought_probs, k=5)
            top_k_data = [
                {"token": tokenizer.decode(top_k_indices[j].item()),
                 "prob": top_k_probs[j].item()}
                for j in range(len(top_k_indices))
            ]
            top_k_thoughts.append(top_k_data)

            # Top-p
            top_p_probs, top_p_indices = topp(thought_probs, p=0.9)
            top_p_data = [
                {"token": tokenizer.decode(top_p_indices[j].item()),
                 "prob": top_p_probs[j].item()}
                for j in range(len(top_p_indices))
            ]
            top_p_thoughts.append(top_p_data)

        if rank == 0:
            print("Legible thoughts (top-5 tokens per latent):")
            print("-" * 80)
            for i, top_k_data in enumerate(top_k_thoughts):
                print(f"Latent thought {i+1}:")
                for item in top_k_data:
                    print(f"  {item['token']}: {item['prob']*100:.1f}%")
                print()

            print("-" * 80)
            print("Legible thoughts (top-p=0.9 tokens per latent):")
            print("-" * 80)
            for i, top_p_data in enumerate(top_p_thoughts):
                print(f"Latent thought {i+1} (top-p=0.9, {len(top_p_data)} tokens):")
                for j, item in enumerate(top_p_data[:20]):
                    print(f"  {item['token']}: {item['prob']*100:.1f}%")
                if len(top_p_data) > 20:
                    print(f"  ... and {len(top_p_data) - 20} more tokens")
                print()

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract answer and reasoning
        answer_output = output_text.split("#")[-1].replace(",", "").strip()
        # Get the reasoning (everything before the final answer)
        reasoning = output_text.split("#")[0].strip() if "#" in output_text else ""
        # Remove the question from reasoning
        if question in reasoning:
            reasoning = reasoning.replace(question, "").strip()

        is_correct = answer_output == ground_truth_answer

        if rank == 0:
            print("-" * 80)
            print(f"Model output:\n{output_text}")
            print()
            print(f"Extracted answer: {answer_output}")
            print(f"Correct: {is_correct}")
            print()

            # Store result
            result = {
                "question_idx": idx,
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "model_reasoning": reasoning,
                "model_answer": answer_output,
                "is_correct": is_correct,
                "raw_continuous_thoughts": raw_thoughts,
                "top_k_thoughts": top_k_thoughts,
                "top_p_thoughts": top_p_thoughts
            }
            results.append(result)

    # Save results to JSON file
    if rank == 0:
        output_file = "legibility_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*80}")
        print(f"Results saved to {output_file}")
        print(f"{'='*80}")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()