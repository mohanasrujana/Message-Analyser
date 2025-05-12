import pytest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from scripts.api.pipeline.Models import Model
import requests

@patch("scripts.api.pipeline.Models.subprocess.run")
@patch("scripts.api.pipeline.Models.requests.get")
@patch("scripts.api.pipeline.Models.subprocess.Popen")
def test_load_model_success(mock_popen, mock_get, mock_run):
    mock_get.side_effect = [requests.ConnectionError(), MagicMock(status_code=200)]

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None 
    mock_proc.terminate = MagicMock()
    mock_popen.return_value = mock_proc

    model = Model("mistral")
    model.load_model()

    mock_popen.assert_called_once_with(["ollama", "serve"], stdout=-3, stderr=-3)
    mock_run.assert_called_once_with(["ollama", "pull", "mistral"], check=True)

@patch("scripts.api.pipeline.Models.requests.get")
@patch("scripts.api.pipeline.Models.subprocess.Popen")
def test_load_model_timeout(mock_popen, mock_get):
    mock_get.side_effect = requests.ConnectionError()

    model = Model("mistral")
    with pytest.raises(RuntimeError, match="Ollama server did not start in time."):
        model.load_model()

def test_stop_ollama_terminates_running_process():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.terminate = MagicMock()

    model = Model("mistral")
    model.ollama_proc = mock_proc
    model.stop_ollama()

    mock_proc.terminate.assert_called_once()

def test_stop_ollama_does_nothing_if_already_stopped():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1 

    model = Model("mistral")
    model.ollama_proc = mock_proc
    model.stop_ollama()

    mock_proc.terminate.assert_not_called()