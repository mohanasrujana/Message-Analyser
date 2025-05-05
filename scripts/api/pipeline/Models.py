import subprocess
import atexit
import time
import requests
class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.ollama_proc = None


    def load_model(self):
        # Start the Ollama server
        self.ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        atexit.register(self.stop_ollama)

        # Wait until Ollama is ready
        for _ in range(30):  # wait up to ~15 seconds
            try:
                r = requests.get("http://localhost:11434")
                if r.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Ollama server did not start in time.")

        # Pull the model
        print(f"Pulling model: {self.model_name}")
        subprocess.run(["ollama", "pull", self.model_name], check=True)
        print(f"Model '{self.model_name}' pulled successfully.")


    def stop_ollama(self):
        if self.ollama_proc and self.ollama_proc.poll() is None:
            print("Stopping Ollama server.")
            self.ollama_proc.terminate()
