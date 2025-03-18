import pandas as pd
import random

def process_conversations(file_path, conv_column, label, preprocess_fn):
    """Extract complete conversations from a CSV file
    label for true positive is 1, true negative is 0, ambiguous is 2
    Returns list of dictionaries with 'conversation' and 'label'"""
    df = pd.read_csv(file_path)
    conversations = []
    
    for idx, row in df.iterrows():
        raw_text = row[conv_column]
        processed_text = preprocess_fn(raw_text)
        
        conversations.append({
            'conversation': processed_text,
            'label': f'{label}'
        })
    
    return conversations

def preprocess_file(text):
    """Handles formatting"""
    text = text.strip('"')
    start = text.find('```plaintext') + len('```plaintext')
    end = text.find('```', start)
    return text[start:end].strip()


# Extract data from true negative
file1_conv = process_conversations(
    'true_negative dataset/augmented_true_negative_conversations.csv', 
    'conversation', 
    0, 
    preprocess_file
)

#Extract data from ambiguous
file2_conv = process_conversations(
    'ambiguous_conversations.csv', 
    'conversation', 
    2, 
    preprocess_file
)
'''
TODO: Extract and Combine true positive data

'''

# Combine and shuffle
all_conversations = file1_conv + file2_conv
random.shuffle(all_conversations)

# Add sequential IDs
for idx, conv in enumerate(all_conversations, start=1):
    conv['id'] = idx

# Create DataFrame and save
output_df = pd.DataFrame(all_conversations)
output_df = output_df[['id', 'conversation', 'label']]  # Reorder columns
output_df.to_csv('combined_conversations.csv', index=False)