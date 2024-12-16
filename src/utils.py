import torch
from esm import pretrained

def load_esm_model():
    # Load a smaller ESM2 model for embeddings
    esm_model, alphabet = pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    return esm_model, batch_converter

def get_esm_embedding(sequence, esm_model, batch_converter):
    """Generate ESM embedding for a single sequence."""
    # Handle None sequences by returning zero embedding
    if sequence is None:
        return torch.zeros(320)

    data = [("", sequence)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
    embedding = results["representations"][6].mean(dim=1).squeeze(0)  # Mean pooling
    return embedding
