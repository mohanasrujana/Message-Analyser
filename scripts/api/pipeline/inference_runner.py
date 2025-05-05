import pandas as pd
import time
import csv
import logging
from pathlib import Path
from functools import lru_cache
import ollama
from pipeline.Models import Model
from flask_ml.flask_ml_server.models import (
    FileResponse, FileType
)


logger = logging.getLogger(__name__)

def predict(model_name: str, prompt: str) -> str:
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def build_prompt(conversation: str, use_case: str, custom_prompt: str = None) -> str:
    """ .     
    Build the prompt based on the use case and provided conversation.
    
    Args:
        conversation (str): The conversation text in [Message i] Name: Content format.
        use_case (int): 1 for Actus Reus, 2 for Mens Rea, 3 for Custom Prompt.
        custom_prompt (str, optional): The user-defined prompt for use case 3.

    Returns:
        str: Final prompt ready for model input.
    """
    
    if use_case == '1':
        crime_element = "Actus Reus"
        definition = '''**Definitions:**

        * **Actus Reus (Guilty Act):** This refers to the physical act of committing a crime. It's the tangible, observable action that constitutes the criminal offense.'''
        no_element_response = "No. There is no element of Actus Reus in the conversation."
        gold_example = """
        GOLD EXAMPLE:

        Conversation:
        [Message 1] Liam: I smashed the window and climbed inside.
        [Message 2] Rachel: I grabbed the jewelry and ran out.
        [Message 3] Emily: Did you see the new bakery opening downtown?

        Expected Output:
        Yes. Evidence:
        Actus Reus:
        [Message 1 - Name]: I smashed the window and climbed inside.
        [Message 2 - Name]: I grabbed the jewelry and ran out.
        """
    
    elif use_case == '2':
        crime_element = "Mens Rea"
        definition = '''* **Mens Rea (Guilty Mind):** This refers to the mental state of the perpetrator at the time the crime was committed. It encompasses the intention, knowledge, or recklessness that the person had when performing the act. In essence, it's about proving that the person knew what they were doing was wrong.'''
        no_element_response = "No. There is no element of Mens Rea in the conversation."
        gold_example = """
        GOLD EXAMPLE:

        Conversation:
        [Message 1] Liam: We should plan the robbery carefully.
        [Message 2] Rachel: I'll study the security system tonight.
        [Message 3] Emily: Are you coming to the party tomorrow?

        Expected Output:
        Yes. Evidence:
        Mens Rea:
        [Message 1 - Name]: We should plan the robbery carefully.
        [Message 2 - Name]: I'll study the security system tonight.
        """
    
    elif use_case == '3':
        crime_element = custom_prompt if custom_prompt else "Relevant Messages"
        definition = ""
        no_element_response = "No. There is no message that matches the prompt in the given conversation."
        gold_example = f"""
        GOLD EXAMPLE:

        USER INSTRUCTION:
        Find messages showing distrust or suspicion.

        Conversation:
        [Message 1] Liam: Are you sure this plan will work?
        [Message 2] Rachel: I don't trust him with the money.
        [Message 3] Emily: Can't wait for the vacation next week!

        Expected Output:
        Yes. Evidence:
        Distrust or Suspicion:
        [Message 1 - Name]: Are you sure this plan will work?
        [Message 2 - Name]: I don't trust him with the money.
        """
    else:
        raise ValueError(f"Invalid use_case: {use_case}. Must be 1 (Actus Reus), 2 (Mens Rea), or 3 (Custom Prompt).")
    
    prompt = f"""
    You are a forensic conversation analyst specializing in detecting **{crime_element}** from chat conversations.

    Your job is to carefully read the conversation and extract only the messages that match the definition of {crime_element}.
    {definition}
    ---

    DO NOT:
    - Do NOT paraphrase, summarize, or explain.
    - Do NOT guess or assume hidden meanings.
    - Do NOT include observations, interpretations, or conclusions.
    - ONLY output if there is clear evidence matching {crime_element}.

    ---

    STRICT OUTPUT FORMAT:
    Yes. Evidence:
    {crime_element}:
    [Message 1 - Name]: <exact message text>
    [Message 2 - Name]: <exact message text>

    If no relevant messages are found, output exactly:
    {no_element_response}

    ---

    INCORRECT FORMATS (DO NOT DO THIS):
    - {crime_element}: They probably meant something criminal.
    - {crime_element}: Someone sounded suspicious.
    - Any summaries, bullet points, or assumptions.

    {gold_example}

    ---

    Now analyze the following conversation:

    {conversation}

    ---
    Extract your output below:
    """
    
    return prompt


def process_conversations(model_name,list_of_conversations,usecase,custom_prompt="") -> list:
    
    engine = Model(model_name)
    engine.load_model()
    
    list_of_raw_outputs = []
    
    for i in range(len(list_of_conversations)):
        logger.info(f"Processing conversation {i+1}/{len(list_of_conversations)}")
        print(f"Processing conversation {i+1}/{len(list_of_conversations)}")
        prompt = build_prompt(list_of_conversations[i],usecase)
        model_raw_output = predict(model_name,prompt)
        list_of_raw_outputs.append(model_raw_output)
        logger.info(f"Completed conversation {i+1}")
        print(f"Completed conversation {i+1}")

    return list_of_raw_outputs
