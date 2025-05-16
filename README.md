# Message-Analyzer

**Message Analyzer** is a forensic tool that analyzes text conversations to extract key criminal elements, with a special focus on murder-related analysis for now. The system uses **Gemma 3** and **Mistral 7B** model and is deployed via a structured Flask-ML API.


The project includes:
- **Conversation-Based Crime Analysis**
- **Criminal Element Extraction based on the usecase**
- **Flask-ML API and Command Line Interface (CLI) support**
- **Ollama Inference: Uses Mistral 7B and Gemma 3 locally**
- **Clean Output with structured results**

---

## Getting Started

### Prerequisites
- Python **>=3.8**
- Flask
- FlasMl
- numpy, pandas, scikit_learn, torch
- fpdf, pdfplumber
- Pip **(latest version recommended)**
- Ollama installed locally and running
- Mistral 7B and Gemma 3 model pulled via:
```bash
ollama pull mistral:7b-instruct
ollama pull gemma3:12b
```

### Clone the Repository
```bash
git clone https://github.com/mohanasrujana/Message-Analyser.git
cd Message-Analyzer
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Key Components

### **Command Line Interface (CLI) (`cli.py`)**
- Provides an easy-to-use CLI for message analyzisation.

### **Flask-ML API (`server.py`)**
- Deploys the model as a web API.

---


## Running Inference

### CLI Usage

CLI help

``` bash
python -m scripts.api.cli --help  
```

Predicting the conversations using cli:

Replace conversations_file with the input file of your conversations, results_dir with the directory in which you want your results, and "Actus Reus,Mens Rea" with the elements of crime you'd like to extract
```bash
python -m scripts.api.cli analyze \         
  --input_file <Comma separated files> \   
  --output_dir <output directory> \  
  --model_name <MISTRAL7B/GEMMA3> \   
  --output_type <csv/pdf/tx/xlsx> \                                                                       
  --usecase < 1 for ActusReus , 2 for MensRea, 3 for CustomPrompt>
  --usecase3 <prompt, give input here if you are choosing usecase 3, otherwise u dont need to give any input here> 
```

Here's the example command that worked for us:
```bash
python -m scripts.api.cli analyze \
  --input_file "/Users/satyasrujanapilli/Message-Analyser/test_conversation - Sheet1.csv" \
  --output_dir "/Users/satyasrujanapilli/Message-Analyser/results" \
  --model_name MISTRAL7B \
  --output_type csv \
  --usecase 3 \
  --usecase3 "Does this conversation have evidence tampering"
```



### API Usage
Start the Flask-ML API server:
```bash
python -m scripts.api.server
```

#### Server usage(method 2)
##### Use Rescue-Box-Desktop

- Install Rescue-Box from [link](https://github.com/UMass-Rescue/RescueBox-Desktop)
- Open Rescue-Box-Desktop and register the model by adding the server IP address and port number in which the server is running.
- Give all the input files or a single file and select a destination directory for output. 
- Choose the model from list of available models under the **MODELS** tab.
- Select the analysis you want to do (Actus Reus, Mens Rea, Custom Prompt) 
Tip : Input N/A if you are using Actus Reus or Mens Rea because all the fields are required. 
- Choose the output file type for the file format you want the output to be.
- Run the model. 
- View the output in Jobs
- Click on view to view the details and results

---
## Output Format
conversation_id | UseCase | Analyzed Messages

---
## Limitations
1. Can handle conversations less than 60 messages only.
2. Format types like pdf/txt are not being displayed on rescuebox even though the file is being generated in the output directory. Talked to Prasanna and got to know that RescueBox can't handle these formats.
---

## Future Enhancements

1. ** Apply this to child rescue **
2. ** Add audio support for phone calls and voice notes** 
3. ** Integrate models with larger context window**

---

## Authors

1. Satya Srujana Pilli

2. Ashwini Ramesh Kumar 

3. Shalom Jaison

---




