import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from coconut import Coconut

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
    saved_weights = torch.load("path/to/checkpoint", map_location=f"cuda:{local_rank}")
    model.load_state_dict(saved_weights, strict=False)

    # Move to GPU and set eval mode
    model = model.to(local_rank)
    model.eval()

    # Tokenize prompt
    prompt = "What is the sum of 43 and 12?\n<|start-latent|><|latent|><|latent|><|latent|><|end-latent|>"
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

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if rank == 0:
        print(output_text)
        print(len(latent_hidden_states))

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()