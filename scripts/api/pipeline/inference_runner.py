import pandas as pd
import os
import time
import csv
import logging
from pathlib import Path
from model_registry import load_model
from functools import lru_cache
from flask_ml.flask_ml_server.models import (
    FileResponse, FileType
)


logger = logging.getLogger("inference-runner")


@lru_cache(maxsize=1)
def get_inference_engine():
    model_name = os.environ.get("MODEL_NAME", "gemma")
    return load_model(model_name)

def process_conversations(df: pd.DataFrame, output_path: str, crime_elements:str) -> FileResponse:
    start_time = time.time()
    engine = get_inference_engine()
    results = []
    RESULTS_DIR = Path(output_path)
    RESULTS_FILE = RESULTS_DIR / "predicted_result.csv"

    for i, row in df.iterrows():
        conversation = row["conversation"]
        logger.info(f"Processing conversation {i+1}/{len(df)}")

        messages = conversation.strip().split("\n")
        # chunks = [messages[k:k + 30] for k in range(0, len(messages), 30)]
        # Extract criminal activity
        chunk_text = "\n".join([m.strip().replace("\r", "") for m in messages])
        raw_output = engine.extract_criminal_activity(chunk_text, crime_elements)
        print(raw_output)
        grouped_result = engine.parse_results_grouped(raw_output, conversation_id=i+1, chunk_id=i+1)
        for crime_element, messages in grouped_result.items():
            for res in messages:
                res["crime_element"] = crime_element
                results.append(res)
        # for j, chunk in enumerate(chunks):
        #     chunk_text = "\n".join(chunk)
        #     raw_output = engine.extract_criminal_activity(chunk_text, crime_elements)
        #     logger.info(f"Raw output: {raw_output}")

        #     grouped_result = engine.parse_results_grouped(raw_output, conversation_id=i+1, chunk_id=j+1)
        #     for crime_element, messages in grouped_result.items():
        #         for res in messages:
        #             res["crime_element"] = crime_element
        #             results.append(res)
            
        logger.info(f"Completed conversation {i+1}, found {len(results)} relevant messages")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["conversation_id", "chunk_id", "message_number", "speaker", "message", "crime_element"])
        writer.writeheader()
        writer.writerows(results)
    
    # Return the file response
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s. Results saved to {RESULTS_FILE}")
    return FileResponse(path=str(RESULTS_FILE), file_type=FileType.CSV)
