import torch
from transformers import AutoModel, AutoFeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/wav2vec2-base-960h"

print("Loading model...")
model = AutoModel.from_pretrained(model_name).to(device).float()
model.eval()

# Simulate a batch with 8 audio files, 1 is long (e.g., 500 frames), 7 are short (e.g., 100 frames)
# 500 frames ~ 160000 samples. 100 frames ~ 32000 samples.
input_values = torch.zeros(8, 160000).to(device)
input_values[0, :] = torch.randn(160000).to(device) * 0.1
for i in range(1, 8):
    input_values[i, :32000] = torch.randn(32000).to(device) * 0.1
    # the rest is exactly 0.0

attention_mask = torch.zeros(8, 160000, dtype=torch.long).to(device)
attention_mask[0, :] = 1
for i in range(1, 8):
    attention_mask[i, :32000] = 1

print("Forward pass...")
with torch.no_grad():
    outputs = model(input_values, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state

nans = hidden_states.isnan().sum().item()
print(f"Total NaNs in hidden_states: {nans}")

# What if without attention_mask?
with torch.no_grad():
    outputs2 = model(input_values, attention_mask=None)
    hidden_states2 = outputs2.last_hidden_state
nans2 = hidden_states2.isnan().sum().item()
print(f"Total NaNs without attention_mask: {nans2}")
