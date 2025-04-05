from .mistral_ollama import MistralOllamaInference

def load_model(model_name: str = "mistral"):
    if model_name == "mistral":
        return MistralOllamaInference()
    else:
        raise ValueError(f"Unknown Model Name: {model_name}")