import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")

def compute_and_save_subprompt_embeddings(subprompts_by_eoc, output_path="data/subprompt_embeddings.pkl"):
    embeddings = {
        eoc: model.encode(prompts, convert_to_tensor=True)
        for eoc, prompts in subprompts_by_eoc.items()
    }
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

def load_subprompt_embeddings(path="data/subprompt_embeddings.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)