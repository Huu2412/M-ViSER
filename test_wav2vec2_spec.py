import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("facebook/wav2vec2-base", apply_spec_augment=True)

# Let's see if apply_spec_augment is an attribute of the encoder itself
has_attr = hasattr(model.encoder, "apply_spec_augment")
print("model.encoder has apply_spec_augment:", has_attr)
if has_attr:
    print("Value:", model.encoder.apply_spec_augment)

# How about the wav2vec2 encoder inside
if hasattr(model, "encoder") and hasattr(model.encoder, "apply_spec_augment"):
    print("Value in model.encoder:", model.encoder.apply_spec_augment)
