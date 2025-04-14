from .mistral_ollama import MistralOllamaInference
from .gemma_ollama import GemmaOllamaInference

def load_model(model_name: str = "gemma"):
    if model_name == "mistral":
        return MistralOllamaInference()
    elif model_name == "gemma":
        return GemmaOllamaInference()
    else:
        raise ValueError(f"Unknown Model Name: {model_name}")