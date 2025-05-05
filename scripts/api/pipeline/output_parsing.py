import re
from typing import List, Dict
from pathlib import Path
import pandas as pd
from fpdf import FPDF
import unicodedata

class OutputParser:
    def __init__(self, output_dir: str, output_file_type: str):
        self.output_dir = output_dir
        self.output_file_type = output_file_type
        self.result_dir = Path(output_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def process_raw_output(self, raw_output_list):
        results = []
        for i, raw_output in enumerate(raw_output_list):
            print(raw_output)
            curr_result = self.parse_results_grouped(raw_output, conversation_id=i+1, chunk_id=i+1)
            print(curr_result)
            results.append(curr_result)

        self.save_to_file(results)
    
    def parse_results_grouped(self, model_output: str, conversation_id: int, chunk_id: int) -> Dict[str, List[Dict]]:
        """
        Parses the model output and returns a dictionary grouping messages by crime element.
        """
        result = {"Answer": None, "Evidence": {"category": None, "message_text": None}}
        lines = model_output.strip().splitlines()
        if not lines:
            return result

        match_answer = re.match(r"^(Yes|No)\.(?:\s+Evidence:)?(.*)?$", lines[0], re.IGNORECASE)
        if not match_answer:
            return result
        
        answer = match_answer.group(1)
        evidence = match_answer.group(2).strip()

        if answer.lower() == "yes":
            evidence = []

            for line in lines[1:]:
                stripped_line = line.strip()
                if stripped_line:
                    category_match = re.match(r"^(.*?):$", stripped_line)
                    message_match = re.match(r"^\[?Message\s+(\d+)\s*[-:]?\s*(.*?)\]?:?\s*(.*)", stripped_line, re.IGNORECASE)

                    if category_match: # This essentially stores the prompt/category of the messages [Actus Reus, Mens Rea]
                        result["Evidence"]["category"] = category_match.group(1)
                    elif message_match:
                        evidence.append(stripped_line)
        else:
            evidence = [evidence if evidence else "\n".join([line.strip() for line in lines[1:] if line.strip()])]
        
        result["Answer"] = answer
        result["Evidence"]["message_text"] = evidence
        return result


    def clean_text_for_pdf(self, text: str) -> str:
        """
        Converts fancy unicode characters to closest ASCII equivalents.
        """
        if not isinstance(text, str):
            return text
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    def save_to_file(self, results: List[Dict]):
        rows = []
        for i, result in enumerate(results):
            conv_id = i + 1
            answer = result["Answer"]
            evidence = result["Evidence"]
            category = evidence["category"] if evidence else None
            messages = evidence["message_text"]
            
            if isinstance(messages, list):
                joined_messages = "\n".join(messages)
            else:
                joined_messages = messages
            rows.append({'conversation_id': conv_id, 
                         'category': category,
                         'message_text': joined_messages})
        
        df = pd.DataFrame(rows)
        
        output_base = self.result_dir / "parsed_output"
        if self.output_file_type.lower() == "csv":
            df.to_csv(output_base.with_suffix('.csv'), index=False)
        elif self.output_file_type.lower() == "xlsx":
            df.to_excel(output_base.with_suffix('.xlsx'), index=False)
        elif self.output_file_type.lower() == "txt":
            with open(output_base.with_suffix(".txt"), "w", encoding='utf-8') as f:
                for _, row in df.iterrows():
                    f.write(f"{row['conversation_id']} | {row['category']} | {row['message_text']}\n")
        elif self.output_file_type.lower() == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for _, row in df.iterrows():
                clean_message = self.clean_text_for_pdf(row['message_text'])
                pdf.multi_cell(0, 10, f"{row['conversation_id']} | {row['category']} | {row['message_text']}\n")
            pdf.output(str(output_base.with_suffix(".pdf")))
        else:
            raise ValueError(f"Unsupported file type: {self.output_file_type}")

if __name__ == "__main__":
    opt = OutputParser(output_dir="../../../data/test_outputs_parsing", output_file_type="xlsx")

    # Example 1: Proper Yes Evidence
    model_output_yes = """
    Yes. Evidence:
    Distrust or Suspicion:
    Message 1: Are you sure this plan will work?
    Message 2: I don't trust him with the money.
    """

    # Example 2: Proper No Evidence
    model_output_no = """
    No. There is no message that matches the prompt in the given conversation.
    """

    # Example 3: Messy output without colon
    model_output_messy = """
    Yes. Evidence:
    Distrust or Suspicion:
    Message 1 Are you sure this plan will work?
    Message 2:  I don't trust him with the money.
    """

    raw_outputs = [model_output_yes, model_output_no, model_output_messy]
    opt.process_raw_output(raw_outputs)
