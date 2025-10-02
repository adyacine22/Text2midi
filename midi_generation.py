import pickle
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from model.transformer_model import Transformer
from huggingface_hub import hf_hub_download
import gc
import os
import shutil


def generate_midi_optimized(prompt, output_file="output.mid"):
    # Use local artifacts if available, otherwise download
    local_model = "artifacts/pytorch_model.bin"
    local_tokenizer = "artifacts/vocab_remi.pkl"
    
    if os.path.exists(local_model) and os.path.exists(local_tokenizer):
        model_path = local_model
        tokenizer_path = local_tokenizer
    else:
        repo_id = "amaai-lab/text2midi"
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")
        
        # Save to artifacts for next time
        os.makedirs("artifacts", exist_ok=True)
        shutil.copy2(model_path, local_model)
        shutil.copy2(tokenizer_path, local_tokenizer)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load tokenizer and model
    with open(tokenizer_path, "rb") as f:
        r_tokenizer = pickle.load(f)

    vocab_size = len(r_tokenizer)
    print("Vocab size: ", vocab_size)

    # Load model
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Set model to not require gradients at all
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    print("Model loaded.")

    print("Generating for prompt: " + prompt)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Generate with explicit memory management
    with torch.no_grad():
        torch.cuda.empty_cache() if device == "cuda" else None
        torch.mps.empty_cache() if device == "mps" else None

        # Generate in smaller chunks or with explicit memory clearing
        output = model.generate(input_ids, attention_mask, max_len=50, temperature=1.0)

        # Immediately extract the result and clean up
        output_list = output[0].cpu().tolist()  # Move to CPU first

        # Clean up tensors immediately
        del output, input_ids, attention_mask, inputs

    # Clean up model and tokenizer
    del model, tokenizer

    # Aggressive memory cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.empty_cache()

    gc.collect()

    # Create output directory and get incremental filename
    os.makedirs("generated_midi", exist_ok=True)
    counter = 1
    while os.path.exists(f"generated_midi/output_{counter:03d}.mid"):
        counter += 1
    output_path = f"generated_midi/output_{counter:03d}.mid"

    # Generate MIDI
    generated_midi = r_tokenizer.decode(output_list)
    generated_midi.dump_midi(output_path)
    print(f"MIDI saved: {output_path}")

    # Final cleanup
    del r_tokenizer, generated_midi
    gc.collect()

    print("MIDI generated successfully!")


# Usage
if __name__ == "__main__":
    prompt = "create a pop rock music in A minor, with guitar, bass, drums and voice with a 120 bpm tempo and a 4/4 structure"

    generate_midi_optimized(prompt)
