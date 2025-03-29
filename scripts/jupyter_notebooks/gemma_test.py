# -*- coding: utf-8 -*-
"""gemma_test.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kiNBGjHSxFX0sq2AEKRBvogj4Sxr8gdm
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/gemma-2b"

from huggingface_hub import login
login()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate(conversation):
  input_text =  prompt = """Extract messages from the given conversation that contain any indications of possible criminal activity or legal implications. Categorize each extracted message under the appropriate crime element and format the output strictly as:

- Message from <Speaker>: <Message> | Crime element: <Category>

Crime elements include but are not limited to:

- Actus Reus (Guilty Act): Actions that could constitute a crime (e.g., 'I took the money and ran').
- Mens Rea (Guilty Mind): Intent, knowledge, or recklessness (e.g., 'I knew it was illegal, but I did it anyway').
- Concurrence: Connection between intent and action (e.g., 'He planned to rob the store before doing it').
- Causation: How an action led to harm or a crime (e.g., 'If she hadn’t pushed him, he wouldn’t have fallen').
- Attempt: Preparatory acts or failed attempts at a crime (e.g., 'I tried to hack into the system but got locked out').
- Complicity/Conspiracy: Assisting, encouraging, or planning a crime with others (e.g., 'We planned the break-in together').
- Obstruction of Justice: Interfering with investigations or law enforcement (e.g., 'I deleted the emails before they could find them').
- Extenuating Circumstances: Factors that could reduce or increase culpability (e.g., 'I was forced to do it under threat').

Only extract and categorize messages that match the above crime elements. Ignore all unrelated messages.

Conversation:
{}""".format(conversation)
  input_ids = tokenizer(input_text, return_tensors="pt")

  outputs = model.generate(**input_ids)
  print(tokenizer.decode(outputs[0]))

import pandas as pd

df = pd.read_csv("/content/true_positives_conversations")

conversations = df["conversation"].tolist()
ground_truth= df["ground truth"].tolist()

df.columns

generated_by_gemma= []
for num,i in enumerate(conversations):
  generated_by_gemma.append(generate(i))
  print("Generated: \n\n\n\n {}".format(generated_by_gemma[num]))
  print("{} conversation done".format(num))