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
    InputType,
)

from pipeline.MessageAnalyserInputsHandler import MessageAnalyserInputsHandler
from pipeline.miscellaneous import message_analyser_inputs,Message_analyser_parameters,ModelType,Usecases,OutputType,map_outputfiletype_FileType
from pipeline.inference_runner import process_conversations
from pipeline.output_parsing import OutputParser

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    filename = "Message_Analyser_server.log",
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('')

server = MLServer(__name__)

def create_crime_analysis_task_schema() -> TaskSchema:
    """
    Creates a schema for the message analyser rescue box desktop interface and defines UI input fields and parameters.
    
    Returns:
        TaskSchema: A schema defining the required inputs and configurable parameters.
    """
    input_files = InputSchema(
        key="input_files",
        label="Input files containing conversation(s) for analysis.",
        input_type=InputType.BATCHFILE, 
    )

    output_dir = InputSchema(
        key="output_dir",
        label="Path to the output directory.",
        input_type=InputType.DIRECTORY,
    )

    output_type = ParameterSchema(
        key="output_type",
        label="Output type",
        subtitle="Desired output file type",
        value=EnumParameterDescriptor(
            enum_vals=[
                EnumVal(key=mt.value, label=mt.name)  
                for mt in OutputType
            ],
            default = OutputType.csv.value,
        ),
        
    )
    
    prompt_schema = ParameterSchema(
        key="usecase3",
        label="Custom prompt for message analysing",
        subtitle = "Enter a custom prompt to analyse your input conversation(s)",
        value = TextParameterDescriptor(
            default = ""
        )
    )
    

    usecase_schema = ParameterSchema(
        key="usecase",
        label="Usecases of the message analyser",
        subtitle="",
        value=EnumParameterDescriptor(
            enum_vals=[
                EnumVal(key=mt.value, label=mt.name)  
                for mt in Usecases
            ],
            default = Usecases.Custom_prompt_analysis.value,
        ),
        
    )

    model_schema = ParameterSchema(
        key="model_name",
        label="Model to use for analysis",
        subtitle="Choose a model from the list of our models.",
        value=EnumParameterDescriptor(
            enum_vals=[
                EnumVal(key=mt.value, label=mt.name)  
                for mt in ModelType
            ],
            default=ModelType.Gemma3.value,
        ),
    )
   
    return TaskSchema(
        inputs=[input_files, output_dir],  
        parameters=[model_schema, usecase_schema, output_type, prompt_schema]
    
    )


server.add_app_metadata(
    name="Message Analyzer for Criminal Activity Extraction from Conversations",
    author="Satya Srujana Pilli, Shalom Jaison, Ashwini Ramesh Kumar",
    version="1.0.0",
    info="This application analyses a conversation for the presence of crime elements such as Mens Rea/ Actus Rea and custom prompts.")


@server.route("/analyze", task_schema_func=create_crime_analysis_task_schema, short_title=" Message Analysis", order=0)
def analyze_conversations(inputs: message_analyser_inputs , parameters:Message_analyser_parameters ) -> ResponseBody:
   
    temp_file = None
    output_base = None
    try:
        
        handler = MessageAnalyserInputsHandler(inputs)
        list_of_conversations = handler.load_conversations() 
        
        usecase = parameters.get("usecase", Usecases.Actus_Reus_analysis.value)
        output_dir = inputs["output_dir"]
        model_name = parameters.get("model_name", ModelType.Gemma3.value)
        output_type = parameters.get("output_type", OutputType.csv.value)
        prompt = parameters.get("usecase3", "")
        if prompt == "empty":
            prompt = ""
        print("completed loading")
        start_time = time.time()
        
        list_of_raw_outputs = process_conversations(model_name, list_of_conversations, usecase, prompt) 
        outputs = OutputParser(output_dir,output_type)
        outputs.process_raw_output(list_of_raw_outputs)
        logger.info(f"Analysis completed in {time.time() - start_time:.2f}s.")
        
        output_base = Path(output_dir.path)/ "analysis_of_conversations"
        file_response = FileResponse(path=str(output_base), file_type=map_outputfiletype_FileType(output_type))
        
        return ResponseBody(file_response)
    
    except Exception as e:
        logger.error(f"Error analyzing conversations: {str(e)}")
        error_file = Path("error_log.txt")
        if output_base:
            error_file = output_base / "error_log.txt"
        with open(error_file, "w") as f:
            f.write(f"Error: {e}\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(traceback.format_exc())
        file_response = ResponseBody(FileResponse(path=str(error_file), file_type=FileType.TEXT))
        return file_response
    finally:
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

if __name__ == "__main__":
    print("Starting Message Analysis Server...")
    server.run(host="127.0.0.1", port=5000)
