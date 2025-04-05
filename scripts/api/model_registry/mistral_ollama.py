import ollama
from collections import defaultdict
import re
from typing import List, Dict, Any, TypedDict

class MistralOllamaInference:
    """
    Inference engine using Ollama with Mistral 7B for text generation
    """
    def __init__(self, model_name="mistral:7b-instruct"):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        try:
            print(prompt)
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            # logger.error(f"Ollama generation error: {str(e)}")
            return f"Error: {str(e)}"
    
    def extract_criminal_activity(self, conversation: str, crime_elements_str: str) -> str:
        """
        Extracts specified crime elements from a conversation using Ollama (Mistral).
        Ensures structured output only — no summaries, no interpretations.
        """

        crime_elements = [elem.strip() for elem in crime_elements_str.split(",") if elem.strip()]
        
        gold_examples = """
        Mens Rea:
        {Message Number: 31, Speaker: Marcus, content: "Definitely didn't intend any harm to 5. Corrections Officer."}
        {Message Number: 79, Speaker: Liam, content: "Maybe Marcus is just trying to shift the blame since he and 5. Corrections Officer argued about art."}

        Actus Reus:
        {Message Number: 48, Speaker: Marcus, content: "Just... some guys moving stuff. Quickly."}
        {Message Number: 70, Speaker: Marcus, content: "I'll handle the cleanup. Just… drop it."}
        """

        prompt = f"""
        You are a forensic conversation analyst. Your job is to extract ONLY the following crime elements from the conversation:

        CRIME ELEMENTS TO EXTRACT: {', '.join(crime_elements)}

        ---

        🧠 DO NOT:
        - Do NOT paraphrase, summarize, or explain.
        - Do NOT include bullets or markdown.
        - Do NOT include observations or conclusions.
        - Do NOT output if you're unsure — just leave that element blank.
        - If no message fits, return NOTHING.

        ---

        ✅ OUTPUT FORMAT:
        [List only the requested crime elements.]

        Mens Rea:
        {{Message Number: <num>, Speaker: <name>, content: "<exact message text>"}}

        Actus Reus:
        {{Message Number: <num>, Speaker: <name>, content: "<exact message text>"}}

        ❌ INCORRECT FORMATS:
        - Mens Rea: They sounded guilty.
        - Actus Reus: Someone did something illegal.

        ---

        📌 EXAMPLE:
        {gold_examples}

        ---

        Now analyze the following conversation:

        {conversation}

        ---

        EXTRACTED CRIMINAL ELEMENTS:
        """
        
        return self.generate_text(prompt)

    def parse_results_grouped(self, model_output: str, conversation_id: int, chunk_id: int) -> Dict[str, List[Dict]]:
        """
        Parses the model output and returns a dictionary grouping messages by crime element.
        """
        grouped_results = defaultdict(list)
        current_crime_element = None

        for line in model_output.splitlines():
            line = line.strip()
            if not line:
                continue

            # Detect crime element section
            crime_element_match = re.match(r"^(Mens Rea|Actus Reus|Concurrence|Causation|Attempt|Complicity/Conspiracy|Obstruction of Justice|Extenuating Circumstances):$", line)
            if crime_element_match:
                current_crime_element = crime_element_match.group(1)
                continue

            # Match structured message line
            message_match = re.match(r"\{Message Number:\s*(\d+),\s*Speaker:\s*(.*?),\s*content:\s*\"(.*?)\"\}", line)
            if message_match and current_crime_element:
                message_number = int(message_match.group(1))
                speaker = message_match.group(2).strip()
                content = message_match.group(3).strip()

                grouped_results[current_crime_element].append({
                    "conversation_id": conversation_id,
                    "chunk_id": chunk_id,
                    "message_number": message_number,
                    "speaker": speaker,
                    "message": content
                })

        return dict(grouped_results)
