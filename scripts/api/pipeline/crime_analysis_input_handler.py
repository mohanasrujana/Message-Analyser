import logging
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from flask import request
from flask_ml.flask_ml_server.models import FileInput, BatchFileInput

logger = logging.getLogger("file-loader")

class CrimeAnalysisInputsHandler:
    """
    Handles single or batch file inputs for crime analysis.
    Reads CSV files from FileInput or BatchFileInput and concatenates them into a single DataFrame.
    """
    def __init__(self, inputs: dict):
        """
        Initializes the handler with the provided inputs.

        Args:
            inputs (dict): A dictionary containing either:
                - 'input_file': FileInput
                - 'input_files': BatchFileInput
        """
        self.inputs = inputs
        # Gather FileInput objects into a list
        if "input_files" in inputs:
            batch: BatchFileInput = inputs["input_files"]
            self.file_inputs = batch.files
        elif "input_file" in inputs:
            single: FileInput = inputs["input_file"]
            self.file_inputs = [single]
        else:
            raise KeyError("Expected 'input_files' or 'input_file' in inputs")

    def load_input_df(self) -> pd.DataFrame:
        """
        Reads each CSV input into a pandas DataFrame and concatenates them.

        Returns:
            pd.DataFrame: The combined DataFrame containing all rows from the input files.

        Raises:
            ValueError: If a file cannot be processed or if the required 'conversation' column is missing.
        """
        dfs = []
        for file_input in self.file_inputs:
            # Create a temporary file for CSV extraction
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            temp_path = temp.name
            try:
                # 1) Uploaded file object
                if hasattr(file_input, "file"):
                    logger.info("Reading from file_input.file")
                    content = file_input.file.read()
                    temp.write(content)
                    temp.flush()

                # 2) Path on disk
                elif hasattr(file_input, "path"):
                    logger.info(f"Copying from disk path: {file_input.path}")
                    shutil.copyfile(file_input.path, temp_path)

                # 3) Fallback to Flask request.files
                elif hasattr(request, "files") and "input_file" in request.files:
                    logger.info("Reading from Flask request.files['input_file']")
                    uploaded = request.files["input_file"]
                    uploaded.save(temp_path)

                else:
                    raise ValueError("Cannot extract CSV from FileInput object")

                temp.close()
                logger.info(f"Reading CSV into DataFrame: {temp_path}")
                df = pd.read_csv(temp_path)

            except Exception as e:
                logger.error(f"Error extracting CSV: {e}")
                raise ValueError(f"Could not process file {file_input}: {e}")

            finally:
                # Clean up temporary file
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass

            # Validate required column
            if "conversation" not in df.columns:
                raise ValueError(f"CSV {file_input} missing required 'conversation' column")

            dfs.append(df)

        # Concatenate all DataFrames into one
        combined = pd.concat(dfs, ignore_index=True)
        return combined
