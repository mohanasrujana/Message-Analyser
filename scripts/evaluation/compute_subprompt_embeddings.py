import pickle
from sentence_transformers import SentenceTransformer
from subprompts_text import subprompts_by_eoc_generic_prompts, subprompts_by_eoc_real_examples

model = SentenceTransformer("intfloat/e5-large-v2")

# Compute and save generic embeddings
generic_embeddings = {
    eoc: model.encode(prompts, convert_to_tensor=True)
    for eoc, prompts in subprompts_by_eoc_generic_prompts.items()
}
with open("data/subprompt_embeddings_generic.pkl", "wb") as f:
    pickle.dump(generic_embeddings, f)

# Compute and save real example embeddings
real_embeddings = {
    eoc: model.encode(prompts, convert_to_tensor=True)
    for eoc, prompts in subprompts_by_eoc_real_examples.items()
}
with open("data/subprompt_embeddings_real.pkl", "wb") as f:
    pickle.dump(real_embeddings, f)