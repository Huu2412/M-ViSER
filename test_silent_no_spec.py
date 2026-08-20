import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/wav2vec2-base-960h"

print("Loading model...")
model = AutoModel.from_pretrained(
    model_name,
    apply_spec_augment=False,
    attn_implementation="eager"
).to(device).float()
model.train() 

input_values = torch.zeros(2, 16000).to(device)
attention_mask = torch.ones(2, 16000, dtype=torch.long).to(device)

outputs = model(input_values, attention_mask=attention_mask)
nans = outputs.last_hidden_state.isnan().sum().item()
print(f"Total NaNs with apply_spec_augment=False: {nans}")
