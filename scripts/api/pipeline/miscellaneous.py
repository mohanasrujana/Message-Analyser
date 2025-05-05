from typing import TypedDict
from enum import Enum
from flask_ml.flask_ml_server.models import (
     BatchFileInput,
    DirectoryInput,
    FileType
)
class Message_analyser_parameters(TypedDict, total = False):
    model_name : str
    usecase: str
    output_type:str
    usecase3: str

class ModelType(str, Enum):
    Gemma3   = "GEMMA3"
    Mistral7b = "MISTRAL7B"

class message_analyser_inputs(TypedDict):
    input_files : BatchFileInput
    output_dir: DirectoryInput


class Usecases(str, Enum):
    Actus_Reus_analysis = "1"
    Mens_Rea_analysis = "2"
    Custom_prompt_analysis = "3"

class OutputType(str,Enum):
    csv = "csv"
    xlsx = "xlsx"
    pdf = "pdf"
    txt = "txt"

def map_outputfiletype_FileType(output_type):
    if output_type == "csv":
        return FileType.CSV
    elif output_type == "txt":
        return FileType.TEXT