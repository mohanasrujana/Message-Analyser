import logging
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from flask import request

logger = logging.getLogger("file-loader")

def load_input_df(inputs) -> pd.DataFrame:
    """
    Handles various file input cases (direct file, path).
    Extracts the file and returns it as a pandas DataFrame.
    """
    # Debug the input object
    file_input = inputs["input_file"]
    logger.info(f"Input file object type: {type(file_input)}")
    logger.info(f"Input file object attributes: {dir(file_input)}")
    
    # Create a temporary file to work with
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    temp_path = temp_file.name
    logger.info(f"Created temporary file: {temp_path}")
    
    # Try to extract the file content
    try:
        # access the file attribute
        if hasattr(file_input, 'file'):
            logger.info("Using 'file' attribute")
            # Read the file content and write to temp file
            content = file_input.file.read()
            temp_file.write(content)
            temp_file.flush()
        # access the path attribute
        elif hasattr(file_input, 'path'):
            logger.info(f"Using 'path' attribute: {file_input.path}")
            # Copy the file to temp location
            shutil.copyfile(file_input.path, temp_path)
        # Try direct read method
        elif hasattr(file_input, 'read'):
            logger.info("Using 'read' method")
            content = file_input.read()
            temp_file.write(content)
            temp_file.flush()
        # Try accessing the flask request files
        else:
            if hasattr(request, 'files') and 'input_file' in request.files:
                logger.info("Using Flask request.files")
                file = request.files['input_file']
                file.save(temp_path)
            else:
                raise ValueError("Could not extract file content from FileInput object")
                
        # Close the temp file to ensure all data is written
        temp_file.close()
        
        # Now read the CSV from the temporary file
        logger.info(f"Reading CSV from temporary file: {temp_path}")
        df = pd.read_csv(temp_path)     
    except Exception as e:
        logger.error(f"Error extracting file content: {str(e)}")
        raise ValueError(f"Could not process file: {str(e)}")
        
    # Ensure the CSV contains a conversation column
    if "conversation" not in df.columns:
        raise ValueError("CSV file must contain a 'conversation' column")
    
    return df