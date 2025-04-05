from abc import ABC, abstractmethod

class BaseInferenceModel(ABC):
    @abstractmethod
    def extract_criminal_activity(self, conversation: str, crime_elements_str: str) -> str:
        pass

    def parse_results_grouped(self, model_output: str, conversation_id: int, chunk_id: int) -> dict:
        pass