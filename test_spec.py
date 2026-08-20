import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/wav2vec2-base-960h"

print("Loading model...")
model = AutoModel.from_pretrained(
    model_name,
    apply_spec_augment=True,
    attn_implementation="eager"
).to(device).float()
model.train() # SpecAugment only happens in training mode!

for length in [50000, 32000, 16000, 8000, 4000, 2000, 1000]:
    input_values = torch.randn(2, length).to(device) * 0.1
    attention_mask = torch.ones(2, length, dtype=torch.long).to(device)
    
    # Introduce padding in one of the samples to simulate batching
    attention_mask[1, int(length*0.8):] = 0
    input_values[1, int(length*0.8):] = 0
    
    outputs = model(input_values, attention_mask=attention_mask)
    nans = outputs.last_hidden_state.isnan().sum().item()
    print(f"Length {length}: {nans} NaNs")
