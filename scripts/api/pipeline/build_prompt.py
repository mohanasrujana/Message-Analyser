# pipeline/prompt_builder.py
import ollama
from output_parsing import OutputParser
from pathlib import Path 
import pandas as pd

def build_prompt(conversation: str, use_case: int, custom_prompt: str = None) -> str:
    """
    Build the prompt based on the use case and provided conversation.
    
    Args:
        conversation (str): The conversation text in [Message i] Name: Content format.
        use_case (int): 1 for Actus Reus, 2 for Mens Rea, 3 for Custom Prompt.
        custom_prompt (str, optional): The user-defined prompt for use case 3.

    Returns:
        str: Final prompt ready for model input.
    """
    
    if use_case == 1:
        crime_element = "Actus Reus"
        no_element_response = "No. There is no element of Actus Reus in the conversation."
        rule = """
        - ONLY include messages that contain **concrete or likely physical actions** involved in the criminal act."""
        definition = "Actus Reus refers to the physical act of committing a crime, such as theft, assault, tampering, or destruction."
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
        include = f"""
    INCLUDE:
    - Messages describing actual physical actions (e.g., breaking, poisoning, cleaning evidence).
    - Messages about **altering**, **removing**, or **tampering** with objects, locations, or evidence.
    - Messages that describe **carrying out** the plan (not just planning).
    - Any message where the speaker **admits to or describes doing something criminal**, even indirectly.
    - Planning messages (e.g., “Let’s do X”, "We should ....") are not Actus Reus unless the speaker also refers to carrying it out.
    ---
        """
    elif use_case == 2:
        crime_element = "Mens Rea"
        no_element_response = "No. There is no element of Mens Rea in the conversation."
        definition = "Mens Rea refers to the mental state or intent to commit a crime — such as planning, desire, awareness, knowledge of wrongdoing, or willingness to cause harm."
        rule = """
    - Do NOT include emotions, reactions, or vague feelings unless they clearly show intent or knowledge of guilt.
    - ONLY include messages that reflect **intent, knowledge, or motivation** behind the crime.
        """
        include = f"""
    INCLUDE:
    - Messages that indicate intent, desire, or willingness to commit a crime.
    - Messages where the speaker acknowledges guilt, knows the consequences, or tries to hide the truth.
    - Statements showing planning, premeditation, or strategic thinking about a crime.
    - Any message reflecting awareness of wrongdoing or motivation behind an act.
    - Even indirect admissions or references showing mental preparation or willingness to commit the crime should be included.

    Planning a crime, even if not acted upon, **can be Mens Rea**.
    Emotional expressions like fear, regret, or relief do **not count** unless they clearly show knowledge or intent.
    
    ---    
"""
        gold_example = f"""
    GOLD EXAMPLE:

    Conversation:
    [Message 1] Liam: We should plan the robbery carefully.
    [Message 2] Rachel: I'll study the security system tonight.
    [Message 3] Emily: Are you coming to the party tomorrow?

    Expected Output:
    Yes. Evidence:
    Mens Rea:
    [Message 1 - Liam]: We should plan the robbery carefully.
    [Message 2 - Rachel]: I'll study the security system tonight.
    """
    
    elif use_case == 3:
        crime_element = custom_prompt if custom_prompt else "Relevant Messages"
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
        Relevant Messages:
        [Message 1 - Name]: Are you sure this plan will work?
        [Message 2 - Name]: I don't trust him with the money.
        """
    else:
        raise ValueError(f"Invalid use_case: {use_case}. Must be 1 (Actus Reus), 2 (Mens Rea), or 3 (Custom Prompt).")
    
    # Build final prompt
    if use_case == 1 or use_case == 2:
        prompt = f"""
        You are a forensic conversation analyst specializing in detecting {crime_element} from chat conversations.
        {crime_element} Definition: {definition}
        Your job is to carefully read the conversation and extract only the messages that match the definition of {crime_element}.

        ---

        DO NOT:
        - Do NOT paraphrase, summarize, or explain.
        - Do NOT guess or assume hidden meanings beyond what is stated.
        - Do NOT include observations, interpretations, or conclusions.{rule}

        ---
        {include}

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
        
        ---

        {gold_example}

        ---

        Now analyze the following conversation:

        {conversation}

        ---
        Extract your output below:
        """
    elif use_case == 3:
        prompt = f"""
        You are a forensic conversation analyst. Your job is to analyze chat conversations and extract only the messages that match the user’s specific instruction.
        
        ---
        USER INSTRUCTION:
        {custom_prompt}  

        ---

        DO NOT:
        - Do NOT paraphrase, summarize, or explain.
        - Do NOT guess or assume hidden meanings beyond what is stated.
        - Do NOT include interpretations, opinions, or conclusions.
        - ONLY include exact messages that clearly satisfy the user instruction.

        ---

        STRICT OUTPUT FORMAT:
        Yes. Evidence:
        Relevant Messages:
        [Message X - Name]: <exact message text>
        [Message Y - Name]: <exact message text>

        If no relevant messages are found, output exactly:
        No. There is no message that matches the instruction in the given conversation.

        ---

        INCORRECT FORMATS (DO NOT DO THIS):
        - Relevant Messages: They probably meant something related.
        - Relevant Messages: Someone sounded off.
        - Any summaries, assumptions, or paraphrased content.

        {gold_example}

        ---

        Now analyze the following conversation:

        {conversation}

        ---
        Extract your output below:
        ---
        """
    
    return prompt

def generate_text(prompt: str) -> str:
    try:
        print(prompt)
        response = ollama.chat(
            model="gemma3:12b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

    

if __name__ == "__main__":
    opt = OutputParser(output_dir="../../../data/test_outputs_parsing", output_file_type="csv")
    df = pd.read_csv("./true_positives_conversations.csv")
    sample = df.iloc[0]
    config_text = sample["config"]
    conversation_text = sample["conversation"]

    prompt_text = "Find a message where the age is been asked"

    raw_outputs = []
     # --- Use Case 1: Actus Reus ---
    # prompt_1 = build_prompt(conversation_text, use_case=1)
    # model_output_1 = generate_text(prompt_1)
    # raw_outputs.append(model_output_1)

    # --- Use Case 2: Mens Rea ---
    prompt_2 = build_prompt(conversation_text, use_case=2)
    model_output_2 = generate_text(prompt_2)
    raw_outputs.append(model_output_2)

    # # --- Use Case 3: Custom Prompt ---
    # prompt_3 = build_prompt(conversation_text, use_case=3, custom_prompt=prompt_text)
    # model_output_3 = generate_text(prompt_3)
    # raw_outputs.append(model_output_3)

    opt.process_raw_output(raw_outputs)

