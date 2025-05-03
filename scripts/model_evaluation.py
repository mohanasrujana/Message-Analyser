import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score
import ollama
import csv


# def build_eoc_prompt(conversation):
#     return f"""You are an expert legal analyst reviewing an **online chat** between two individuals.

# Your task is to extract only those messages that indicate possible **criminal behavior** or **intent**.

# Specifically, look for:

# - **Actus Reus (Guilty Act)**: Mentions of actions that were part of committing or covering up a crime.
#   - e.g., hiding a body, securing an area, drowning someone, destroying evidence

# - **Mens Rea (Guilty Mind)**: Mentions of intent, planning, deception, fear of being caught, guilt, or regret.
#   - e.g., “we had to do it,” “no one will find out,” “I feel bad about it,” “we planned this”

# ---

# ✅ Format each relevant message **exactly like this**:
# [Message <number> - <Name>]: <message> | Crime element: <Actus Reus or Mens Rea>

# ✅ Only include messages that strongly or subtly indicate one of the above.

# 🚫 If there are no such messages, output exactly:
# No relevant messages found.

# ---

# Conversation:
# {conversation}
# """
    
# def predict_with_gemma3(conversation):
#     prompt = build_eoc_prompt(conversation)
#     try:
#         response = ollama.chat(
#             model="gemma3:12b",  # Adjust name if needed
#             messages=[{"role": "user", "content": prompt}]
#         )
#         print(response)
#         return response['message']['content']
#     except Exception as e:
#         return f"ERROR: {e}"
# df = pd.read_csv("true_positives_conversations.csv")
# df = df.head(5)

# predictions = []

# for row in df.itertuples(index=False):
#     convo_id = row.conversation_id
#     conversation = row.conversation
#     prediction = predict_with_gemma3(conversation)
#     print(prediction)
#     predictions.append((convo_id, prediction))
#     print(f"✅ Predicted conversation {convo_id}")

# # Save predictions
# with open("model_predictions.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["conversation_id", "prediction"])
#     writer.writerows(predictions)

# print("✅ Saved predictions to model_predictions.csv")

N = 1

gt_df = pd.read_csv("ground_truth_labels.csv").head(N)
pred_df = pd.read_csv("model_predictions.csv").head(N)



def extract_message_id(line):
    match = re.search(r"\[Message\s+(\d+)\s*-\s*([^\]]+)\]", line)
    if match:
        return f"Message {match.group(1)} - {match.group(2).strip()}"
    return None

def extract_eoc_lines(text):
    if pd.isna(text) or "no relevant messages found" in text.lower():
        return set()
    lines = text.strip().split('\n')
    return set(line.strip() for line in lines if line.strip())


def extract_label(line):
    match = re.search(r"\|\s*(?:Crime element:\s*)?(Actus Reus|Mens Rea)", line, re.IGNORECASE)
    return match.group(1) if match else None

y_true_all = []
y_pred_all = []

actus_gt_total = 0
actus_pred_correct = 0
mens_gt_total = 0
mens_pred_correct = 0

for i in range(len(gt_df)):
    gt_lines = extract_eoc_lines(gt_df.iloc[i]["ground_truth"])
    pred_lines = extract_eoc_lines(pred_df.iloc[i]["prediction"])

    # Map: Message ID → Label
    gt_map = {extract_message_id(line): extract_label(line) for line in gt_lines}
    pred_map = {extract_message_id(line): extract_label(line) for line in pred_lines}

    # Count GT totals
    for label in gt_map.values():
        if label == "Actus Reus":
            actus_gt_total += 1
        elif label == "Mens Rea":
            mens_gt_total += 1

    # Compare overlaps
    matched_ids = set(gt_map.keys()) & set(pred_map.keys())
    for msg_id in matched_ids:
        if gt_map[msg_id] == pred_map[msg_id]:  # must match both ID and label
            if gt_map[msg_id] == "Actus Reus":
                actus_pred_correct += 1
            elif gt_map[msg_id] == "Mens Rea":
                mens_pred_correct += 1

    # === For Precision/Recall/F1 scoring ===
    # Convert to binary match: set of message_id + label
    gt_binary = set((mid, gt_map[mid]) for mid in gt_map if gt_map[mid] in {"Actus Reus", "Mens Rea"})
    pred_binary = set((mid, pred_map[mid]) for mid in pred_map if pred_map[mid] in {"Actus Reus", "Mens Rea"})

    tp = len(gt_binary & pred_binary)
    fn = len(gt_binary - pred_binary)
    fp = len(pred_binary - gt_binary)

    y_true_all.extend([1] * tp + [1] * fn + [0] * fp)
    y_pred_all.extend([1] * tp + [0] * fn + [1] * fp)

# --- Final Metrics ---
precision = precision_score(y_true_all, y_pred_all)
recall = recall_score(y_true_all, y_pred_all)
f1 = f1_score(y_true_all, y_pred_all)

actus_coverage = (actus_pred_correct / actus_gt_total) * 100 if actus_gt_total else 0
mens_coverage = (mens_pred_correct / mens_gt_total) * 100 if mens_gt_total else 0

# --- Output ---
print(f"\n📊 Gemma3 Evaluation (First {N} Conversations)")
print("-------------------------------------------------")
print(f"Overall Precision:      {precision:.2f}")
print(f"Overall Recall:         {recall:.2f}")
print(f"F1 Score:               {f1:.2f}")
print()
print(f"Actus Reus Coverage:    {actus_coverage:.1f}% ({actus_pred_correct}/{actus_gt_total})")
print(f"Mens Rea Coverage:      {mens_coverage:.1f}% ({mens_pred_correct}/{mens_gt_total})")