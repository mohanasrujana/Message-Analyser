from sentence_transformers import SentenceTransformer, util
from subprompt_embeddings import load_subprompt_embeddings
import torch


model = SentenceTransformer("intfloat/e5-large-v2")  # you can move this to global if reusing

def evaluate_eoc_predictions(predicted_messages, sub_prompts, sub_prompts_embeddings, threshold=0.75):
    """
    Evaluates the model's predicted messages for each specified EOC. 
    
    Args:
    predicted_messages (List[Dict]): Each dict should have a 'message' field (string).
    sub_prompts (List[str]): Representative short prompts describing this EOC.
    threshold (float): Similarity threshold to count a message as valid.

    Returns:
        Dict with precision, avg similarity, valid matches, and detailed scores.
    """
    if not predicted_messages:
        return {
            "precision": 0.0,
            "avg_similarity": 0.0,
            "valid_count": 0,
            "total_predicted": 0,
            "valid_messages": [],
            "per_message_scores": []
        }
    if not torch.is_tensor(sub_prompts_embeddings):
        sub_prompts_embeddings = torch.tensor(sub_prompts_embeddings)
    messages = [m['message'] for m in predicted_messages]
    message_embeddings = model.encode(messages, convert_to_tensor=True)

    valid_count = 0
    similarity_scores = []
    per_message_scores = []
    valid_messages = []

    for idx, message_embedding in enumerate(message_embeddings):
        similarities = util.cos_sim(message_embedding, sub_prompts_embeddings)[0]
        max_score = float(similarities.max())
        best_match_idx = int(similarities.argmax())
        best_prompt = sub_prompts[best_match_idx]

        similarity_scores.append(max_score)

        result = {
            "message": messages[idx],
            "score": max_score,
            "matched_prompt": best_prompt
        }
        per_message_scores.append(result)

        if max_score >= threshold:
            valid_messages.append(predicted_messages[idx])
            valid_count += 1
        
    precision = valid_count / len(predicted_messages)
    avg_similarity = sum(similarity_scores) / len(similarity_scores)

    return {
        "precision": precision,
        "avg_similarity": avg_similarity,
        "valid_count": valid_count,
        "total_predicted": len(predicted_messages),
        "valid_messages": valid_messages,
        "per_message_scores": per_message_scores
    }

def evaluate_all(predictions_by_eoc, subprompts_by_eoc, subprompt_embeddings_by_eoc, threshold=0.75):
    avg_precision = 0
    avg_similarity = 0
    valid_count = 0
    length = 0
    per_eoc_scores = {}


    for eoc in predictions_by_eoc.keys():
        if eoc not in subprompt_embeddings_by_eoc or eoc not in subprompts_by_eoc:
            continue  # skip if either subprompts or embeddings are missing

        predicted = predictions_by_eoc[eoc]
        subprompt_texts = subprompts_by_eoc[eoc]
        subprompt_embeddings = subprompt_embeddings_by_eoc[eoc]

        curr_res = evaluate_eoc_predictions(
            predicted,
            subprompt_texts,
            subprompt_embeddings,
            threshold=threshold
        )        
        per_eoc_scores[eoc] = curr_res
        avg_precision += curr_res["precision"]
        avg_similarity += curr_res["avg_similarity"]
        valid_count += curr_res["valid_count"]
        length += 1
    
    if length > 0:
        avg_precision /= length
        avg_similarity /= length
    else:
        avg_precision = 0.0
        avg_similarity = 0.0
    
    return {
        "avg_precision": avg_precision,
        "avg_similarity": avg_similarity,
        "valid_count": valid_count,
        "per_eoc_scores": per_eoc_scores
    }



