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

    # Process each question in the dataset (first 10 for quick testing)
    for idx, sample in enumerate(gsm_data[:10]):
        question = sample["question"]
        ground_truth_answer = sample["answer"]

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Question {idx+1}/{len(gsm_data)}")
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

        if rank == 0:
            print("Legible thoughts (top-5 tokens per latent):")
            print("-" * 80)

        # Show top-k for each thought
        for i, hidden in enumerate(latent_hidden_states):
            thought_logits = lm_head(hidden)
            thought_probs = torch.softmax(thought_logits, dim=-1)

            top_k_probs, top_k_indices = torch.topk(thought_probs, k=5)

            if rank == 0:
                print(f"Latent thought {i+1}:")
                for j in range(len(top_k_indices)):
                    token_str = tokenizer.decode(top_k_indices[j].item())
                    prob = top_k_probs[j].item()
                    print(f"  {token_str}: {prob*100:.1f}%")
                print()

        if rank == 0:
            print("-" * 80)
            print("Legible thoughts (top-p=0.9 tokens per latent):")
            print("-" * 80)

        # Show top-p for each thought
        for i, hidden in enumerate(latent_hidden_states):
            thought_logits = lm_head(hidden)
            thought_probs = torch.softmax(thought_logits, dim=-1)

            top_p_probs, top_p_indices = topp(thought_probs, p=0.9)

            if rank == 0:
                print(f"Latent thought {i+1} (top-p=0.9, {len(top_p_indices)} tokens):")
                # Show up to 20 tokens to avoid too much output
                for j in range(min(len(top_p_indices), 20)):
                    token_str = tokenizer.decode(top_p_indices[j].item())
                    prob = top_p_probs[j].item()
                    print(f"  {token_str}: {prob*100:.1f}%")
                if len(top_p_indices) > 20:
                    print(f"  ... and {len(top_p_indices) - 20} more tokens")
                print()

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if rank == 0:
            print("-" * 80)
            print(f"Model output:\n{output_text}")
            print()

            # Extract answer
            answer_output = output_text.split("#")[-1].replace(",", "").strip()
            print(f"Extracted answer: {answer_output}")
            print(f"Correct: {answer_output == ground_truth_answer}")
            print()

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()