import sys
import os

# Add project root (i.e., Message-Analyzer/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from subprompts_text import (
    subprompts_by_eoc_generic_prompts,
    # subprompts_by_eoc_real_examples
)
import pickle
from model_eval import evaluate_all
from scripts.api.model_registry import load_model
import pandas as pd
from collections import defaultdict

predictions_by_eoc = defaultdict(list)

df = pd.read_csv("data/combined_conversations_copy.csv")     
target_eocs = "Mens Rea,Actus Reus"
results = []
mistral_model = load_model("mistral")
for i, row in df.iterrows():
        conversation = row["conversation"]

        messages = conversation.strip().split("\n")
        chunks = [messages[k:k + 30] for k in range(0, len(messages), 30)]
        # Extract criminal activity
        for j, chunk in enumerate(chunks):
            chunk_text = "\n".join(chunk)
            raw_output = mistral_model.extract_criminal_activity(chunk_text, target_eocs)

            grouped_result = mistral_model.parse_results_grouped(raw_output, conversation_id=i+1, chunk_id=j+1)
        
            for eoc, msgs in grouped_result.items():
                predictions_by_eoc[eoc].extend(msgs)

# Load embeddings
with open("data/subprompt_embeddings_generic.pkl", "rb") as f:
    embeddings_generic = pickle.load(f)
with open("data/subprompt_embeddings_real.pkl", "rb") as f:
    embeddings_real = pickle.load(f)

# Run evaluations
results_generic = evaluate_all(predictions_by_eoc, subprompts_by_eoc_generic_prompts, embeddings_generic, threshold=0.5)
# results_real = evaluate_all(predictions_by_eoc, subprompts_by_eoc_real_examples, embeddings_real, threshold=0.5)

# Print or compare results
import json
print("📊 GENERIC PROMPTS RESULTS:")
print(json.dumps(results_generic, indent=2, default=str))
