import inspect
from transformers.models.wav2vec2 import modeling_wav2vec2

print("Wav2Vec2Model _mask_hidden_states code:")
print(inspect.getsource(modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states))
