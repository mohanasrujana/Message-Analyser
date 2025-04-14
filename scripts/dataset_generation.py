import requests
import json
import random
import csv
import time
import ollama
import pandas as pd

# locations = [
#     "bar", "alley", "park", "apartment", "car", "rooftop", "warehouse", 
#     "backyard", "motel room", "office"
# ]

# times = ["early morning", "afternoon", "evening", "night", "late night"]

# crime_methods = [
#     "stabbing", "shooting", "poisoning", "strangulation", "bludgeoning", 
#     "drowning", "pushing off a height", "burning", "suffocation", "drug overdose"
# ]

# names = ["John", "Mike", "Sarah", "Emma", "Jake", "Nina", "Tom", "Lily", "Ben", "Rachel", 
#          "Chris", "Amy", "Daniel", "Chloe", "Kevin", "Maya", "Luke", "Anna", "Steve", "Tina"]

# random.shuffle(names)
# name_pairs = [(names[i], names[i+1]) for i in range(0, len(names)-1, 2)]

# message_counts = list(range(30, 65, 5))

# prompt_configs = []

# for i in range(200):
#   participants = random.choice(name_pairs)
#   config = {
#     "conversation_id": i + 1,
#     "location": random.choice(locations),
#     "time": random.choice(times),
#     "crime_method": random.choice(crime_methods),
#     "participants": participants,
#     "num_messages": random.choice(message_counts)
#   }
#   prompt_configs.append(config)

# with open("tp_prompt_configs.json", "w") as f:
#   json.dump(prompt_configs, f, indent=2)

# def build_prompt(config):
#     p1, p2 = config["participants"]
#     prompt = f"""Create an **online chat** between {p1} and {p2}, who are both directly involved in a murder.

# They committed the crime by {config['crime_method']} at a {config['location']} during the {config['time']}.

# The chat should contain exactly {config['num_messages']} messages, formatted like this:
# [Message 1 - {p1}]: <message>
# [Message 2 - {p2}]: <message>
# ...

# Only these two participants should be involved. The conversation must include subtle or explicit indications of the crime, such as intent, planning, or guilt (Actus Reus or Mens Rea).

# Do not include any narration, explanation, or third-party characters"""
#     return prompt

# with open("tp_prompt_configs.json", "r") as f:
#     configs = json.load(f)

# rows = []
# for i, config in enumerate(configs):
#     prompt = build_prompt(config)
#     try:
#         response = ollama.chat(
#             model="mistral",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         conversation = response['message']['content']
#     except Exception as e:
#         print(f"⚠️ Error at conversation {i+1}: {e}")
#         conversation = "ERROR"

#     rows.append([config["conversation_id"], config, conversation])
#     print(f"✅ Conversation {i+1}/200 complete")

#     time.sleep(1.5)  # Add small delay if you're running locally

# # Step 4: Save to CSV
# with open("true_positives_conversations.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["conversation_id", "config", "conversation"])
#     writer.writerows(rows)

# print("✅ All conversations saved to true_positives_conversations.csv")


def extract_ground_truth(conversation):
    extraction_prompt = f"""You are an expert legal analyst reviewing an **online chat** between two individuals.

Your task is to extract only those messages that indicate possible **criminal behavior** or **intent**.

Specifically, look for:

- **Actus Reus (Guilty Act)**: Mentions of actions that were part of committing or covering up a crime.
  - e.g., hiding a body, securing an area, drowning someone, destroying evidence

- **Mens Rea (Guilty Mind)**: Mentions of intent, planning, deception, fear of being caught, guilt, or regret.
  - e.g., “we had to do it,” “no one will find out,” “I feel bad about it,” “we planned this”

---

✅ Format each relevant message **exactly like this**:
[Message <number> - <Name>]: <message> | Crime element: <Actus Reus or Mens Rea>

✅ Only include messages that strongly or subtly indicate one of the above.

🚫 If there are no such messages, output exactly:
No relevant messages found.

---

Conversation:
{conversation}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": extraction_prompt}]
    )
    return response['message']['content']

df = pd.read_csv("true_positives_conversations.csv")
rows = []
for i, row in df.iterrows():
    conversation_id = row['conversation_id']
    conversation = row['conversation']
    
    try:
        extracted = extract_ground_truth(conversation)
    except Exception as e:
        print(f"⚠️ Error in conversation {conversation_id}: {e}")
        extracted = "ERROR"
    
    rows.append([conversation_id, extracted])
    print(f"✅ Ground truth extracted for conversation {conversation_id}")
    time.sleep(1.5)

# Save to CSV
with open("ground_truth_labels.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["conversation_id", "ground_truth"])
    writer.writerows(rows)

print("✅ All ground truths saved to ground_truth_labels.csv")
