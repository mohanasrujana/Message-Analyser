import os
import time
import traceback
import logging
from pathlib import Path
from enum import Enum
from typing import TypedDict
from functools import lru_cache
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    ResponseBody,
    FileResponse,
    FileType,
    InputSchema,
    ParameterSchema,
    TaskSchema,
    TextParameterDescriptor,
    EnumParameterDescriptor,
    EnumVal,
    BatchFileInput,
    DirectoryInput,
    InputType,
)
from model_registry import load_model
#from pipeline.file_loader import load_input_df
from pipeline.crime_analysis_input_handler import CrimeAnalysisInputsHandler
from pipeline.inference_runner import process_conversations
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mistral-server')

# Initialize Flask-ML Server
server = MLServer(__name__)

# Create a singleton instance of the inference engine
@lru_cache(maxsize=1)
def get_inference_engine():
    import os
    model_name = os.environ.get("MODEL_NAME", "mistral")
    return load_model(model_name)

# Define the model type
class ModelType(str, Enum):
    GEMMA3   = "GEMMA3"
    MISTRAL7B = "MISTRAL7B"

# Define input and parameter types for Flask ML
class CrimeAnalysisInputs(TypedDict):
    """
    Defines the expected input structure for the crime analysis task.
    
    Attributes:
        input_files: Paths to the CSV file/files containing conversations.
    """
    input_files: BatchFileInput      # a batch of CSV files
    output_file: DirectoryInput

class CrimeAnalysisParameters(TypedDict):
    """
    Defines the parameters used for crime analysis.
    
    Attributes:
        include_all_messages: Whether to include all messages or only those with crime elements.
    """
    elements_of_crime: str
    model_type: str

# Define the UI schema for the task
def create_crime_analysis_task_schema() -> TaskSchema:
    """
    Creates a schema for the criminal activity extraction task to define UI input fields and parameters.
    
    Returns:
        TaskSchema: A schema defining the required inputs and configurable parameters.
    """
    input_schema = InputSchema(
        key="input_files",
        label="CSV files containing conversations",
        input_type=InputType.BATCHFILE, 
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )
    
    elements_of_crime_schema = ParameterSchema(
        key="elements_of_crime",
        label="Elements of Crime",
        subtitle="Elements of Crime to be found in the Conversation",
        value=TextParameterDescriptor(
            default="Actus Reus,Mens Rea"
        )
    )

    model_schema = ParameterSchema(
        key="model_type",
        label="Model to use for analysis",
        subtitle="Choose GEMMA3 or MISTRAL7B",
        value=EnumParameterDescriptor(
            enum_vals=[
                EnumVal(key=mt.value, label=mt.name)  
                for mt in ModelType
            ],
            default=ModelType.MISTRAL7B.value,
        ),
    )
    
    return TaskSchema(
        inputs=[input_schema,output_schema],  # Only input is the CSV file
        parameters=[elements_of_crime_schema,model_schema]
    )

# Add application metadata
server.add_app_metadata(
    name="Message Analyzer for Criminal Activity Extraction from Conversations",
    author="Satya Srujana Pilli, Shalom Jaison, Ashwini Ramesh Kumar",
    version="1.0.0",
    info="This application extracts and categorizes potential criminal activities from conversation text using a Mistral-7B model."
)


@server.route("/analyze", task_schema_func=create_crime_analysis_task_schema, short_title=" Message Analysis", order=0)
def analyze_conversations(inputs: CrimeAnalysisInputs, parameters: CrimeAnalysisParameters) -> ResponseBody:
    """
    Process a CSV file containing conversations, extract criminal activities, and save results.
    """
    temp_file = None 
    RESULTS_DIR = Path(inputs["output_file"].path)

    try:
        # Extract the csv into a dataframe
        handler = CrimeAnalysisInputsHandler(inputs)
        df = handler.load_input_df()  # DataFrame with a "conversation" column
        #df = load_input_df(inputs)
        crime_elements = parameters.get("elements_of_crime", "Actus Reus,Mens Rea")
        raw_model_type = parameters.get("model_type", ModelType.MISTRAL7B.value).upper()

        try:
            model_type = ModelType(raw_model_type)
        except ValueError:
            raise ValueError(f"model_type must be one of {[m.value for m in ModelType]}")

        file_response = process_conversations(df, inputs["output_file"].path, crime_elements, model_type) 
        return ResponseBody(file_response)
    except Exception as e:
        logger.error(f"Error analyzing conversations: {str(e)}")
        
        # Create error log file
        error_file = Path("error_log.txt")
        if RESULTS_DIR:
            error_file = RESULTS_DIR / "error_log.txt"
        with open(error_file, "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nStack trace:\n")
            f.write(traceback.format_exc())
        file_response = ResponseBody(FileResponse(path=str(error_file), file_type=FileType.TEXT))
        return file_response
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.info(f"Removed temporary file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")

@server.app.route("/", methods=["GET"])
def root():
    """
    Root endpoint to verify that the server is running.
    
    Returns:
        str: A welcome message.
    """
    return "Welcome to the Message Analysis API!"

# Run the server
if __name__ == "__main__":
    print("Starting Message Analysis Server...")
    # Start the server
    server.run(host="127.0.0.1", port=5000)