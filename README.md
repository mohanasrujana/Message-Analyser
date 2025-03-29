# Message-Analyzer

The **Message Analyzer**  aims to  analyze the conversations to find crime related elements specifically murder related. It is built using **Gemma:2b**, exported to **ONNX format** for efficient inference via **ONNXRuntime**.

The project includes:
- **ONNX-based model inference**
- **Command Line Interface (CLI) for easy interaction**
- **API deployment using Flask-ML**

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python **>=3.8**
- Pip **(latest version recommended)**
- ONNXRuntime

### Clone the Repository
```bash
git clone https://github.com/mohanasrujana/Message-Analyzer.git
cd Message-Analyzer
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dataset
Use the conversations in the data folder of the repo.

### Run Model Export (Convert to ONNX)
```bash
python scripts/export_to_onnx.py
```

This will generate ONNX model:
```
models/gemma-2b.onnx
```

---

## Project Structure

```
📂 Message-Analyzer/
│
├── data/
|   ├── processed/
|   |   ├── combined_conversations.csv # Combined Dataset
|   ├── raw/
|   |   ├── true_negative_dataset/ #Prompt variables for true negative conversations
|   |   |   ├── augmented_true_negative_conversations.csv
|   |   |   ├── cities.txt
|   |   |   ├── conversation_topics.txt
|   |   |   ├── cross_validation_statistics.txt
|   |   |   ├── locations.txt
|   |   |   ├── mistral_validation_results.json
|   |   |   ├── participant_ages.txt
|   |   |   ├── participant_genders.txt
|   |   |   ├── participant_interests.txt
|   |   |   ├── participant_occupations.txt
|   |   |   ├── participant_personalities.txt
|   |   |   ├── time_settings.txt
|   |   |   ├── true_negative_conversations.csv
|   |   |   ├── true_negative_permutations.json
|   |   |   ├── true_negative_results.json
|   |   |   ├── true_negative_statistics.txt
|   |   ├── ambiguous_conversations.csv
|   |   ├── true_positives_conversations.csv
|   ├── combined_conversations_copy.csv
|
├── models
|   ├── gemma-2b.onnx # onnx model
|
├── questions
|   ├── investigative_questions.csv
|   ├── preset_questions.json
|
├── results/
|   ├── predicted_result.csv
|   ├── raw_analysis.csv
|
├── scripts/
|   ├── api/
|   |   ├── cli.py
|   |   ├── server_info.md
|   |   ├── server.py
|   ├── jupyter_notebooks/
|   |   ├── Message_Analyser_Message_Generation.ipynb
|   |   ├── Message_Analyser_QuestionGeneration.ipynb
|   |   ├── Message_Analyzer_Ambiguous_Message_Generation.ipynb
|   |   ├── Message_Generation_True_Negative_.ipynb
|   |   ├── Mistral_Eval_True_Positives_.ipynb
|   |   ├── True_positives_ground_truth.ipynb
|   ├── export_to_onnx.py
|   ├── extract_questions.py
|   ├── preprocessing.py
|   ├── run_mistral_inference.py
|   ├── run_onnx_inference.py
|
├── test/
|   ├── model_test.py
|   ├── test_onnx_model.py
|
├── .gitignore
├── requirements.txt  # Dependencies
├── README.md  # Documentation
```

---

## Key Components

### **Model Export Script (`export_to_onnx.py`)**
- Converts the modified **Gemma:2b** model to **ONNX format**.

### **ONNX Inference Script (`run_onnx_inference.py`)**
- Loads the ONNX model and performs inference on a given image.
- Returns **predicted category** and **most probable attributes**.

### **Command Line Interface (CLI) (`cli.py`)**
- Provides an easy-to-use CLI for message analyzisation.

### **Flask-ML API (`server.py`)**
- Deploys the model as a web API.
- Accepts image uploads and returns predictions.

---

## How the Model is Exported to ONNX

The process of exporting the model to ONNX format enables optimized inference using ONNXRuntime and makes the model deployment-ready across various platforms.

---
### **Exporting to ONNX Format**

The PyTorch model is exported to ONNX using the `torch.onnx.export()` function, ensuring compatibility with ONNXRuntime which can be viewed in `scripts/export_to_onnx.py`.

#### **Export Steps:**

Certainly! Here's the text formatted in Markdown:



## Steps to Export the ONNX Model

Exporting the ONNX model involved the following key steps:

### 1. Initialize the Gemma-2B Model

- Load the `google/gemma-2b` model using the Hugging Face Transformers library.
- Set the model to evaluation mode to ensure consistent behavior during export.

### 2. Create a Dummy Input Tensor

- Use the tokenizer to create a dummy input tensor representing a sample input text. This defines the model's input format.

### 3. Export the Model to ONNX Format

- Use the `torch.onnx.export` function to export the model to ONNX format with the following configuration:
  - **Model**: The wrapped Gemma-2B model.
  - **Dummy Input**: The input tensor generated by the tokenizer.
  - **Output File**: `gemma-2b.onnx` in the `models` directory.
  - **Dynamic Axes**: Allow variable batch sizes and sequence lengths.
  - **External Data Format**: Use external data format for large models.
- Use the following code:
```
torch.onnx.export(
            wrapped_model,
            args=tuple(dummy_inputs.values()),
            f=str(onnx_model_path),
            input_names=list(dummy_inputs.keys()),
            output_names=["logits"],
            dynamic_axes={
                **{k: {0: "batch_size", 1: "sequence_length"} for k in dummy_inputs.keys()},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            use_external_data_format=True,
            verbose=False,  # Set to True only for debugging
            keep_initializers_as_inputs=False  # Better for inference optimization
        )
```

The resulting ONNX model is saved as `gemma-2b.onnx` with external data stored in `gemma-2b_data.bin`. This process ensures the model is optimized for deployment using ONNX Runtime.

## Running Inference

### CLI Usage

CLI help

``` bash
python -m scripts.api.cli --help  
```

Predicting the conversations using cli:
Replace conversations_file with the input file of your conversations, analyzed result with the directory in which you want your results. If you want only criminal elements in the messages choose False else True
```bash
python -m scripts.api.cli analyze --input_file [conversations_file] --output_file [analyzed_result] --include_all_messages [true/false]
```

### Running ONNX Inference Directly
```bash
python scripts/run_onnx_inference.py --csv data/raw/true_positives_conversations.csv --output results.csv
```

### API Usage
Start the Flask-ML API server:
```bash
python api/server.py
```
#### Server usage (method 1)
Once running, send a POST request manually on the terminal:
```bash
curl -X POST "http://127.0.0.1:5000/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "input_file=@/path/to/your/conversations.csv" \
     -F "output_file=@/path/to/output/directory" \
     -F "include_all_messages=false"
```

#### Server usage(method 2)
##### Use Rescue-Box-Desktop

- Install Rescue-Box from [link](https://github.com/UMass-Rescue/RescueBox-Desktop)
- Open Rescue-Box-Desktop and resgiter the model by adding the server IP address and port number in which the server is running.
- Choose the model from list of available models under the **MODELS** tab.
- Checkout the Inspect page to learn more about using the model.
- Run the model. 
- View the output in Jobs
- Click on view to view the details and results



---

## Future Enhancements

1. **Increase the prediction accuracy** 

2. **ONNX Quantization**: Optimize model for faster inference.

---




