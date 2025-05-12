import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.api.pipeline.inference_runner import process_conversations, build_prompt, predict


@pytest.fixture
def sample_conversation():
    return """[Message 1] Liam: We should plan the robbery carefully.
[Message 2] Rachel: I'll study the security system tonight.
[Message 3] Emily: Are you coming to the party tomorrow?"""

def test_actus_reus_prompt(sample_conversation):
    prompt = build_prompt(sample_conversation, use_case='1')
    assert "Actus Reus" in prompt
    assert "GOLD EXAMPLE" in prompt
    assert "No. There is no element of Actus Reus in the conversation." in prompt
    assert "[Message 1 - Name]" in prompt

def test_mens_rea_prompt(sample_conversation):
    prompt = build_prompt(sample_conversation, use_case='2')
    assert "Mens Rea" in prompt
    assert "GOLD EXAMPLE" in prompt
    assert "No. There is no element of Mens Rea in the conversation." in prompt
    assert "[Message 1 - Name]" in prompt

def test_custom_prompt(sample_conversation):
    custom_prompt_text = "Find messages showing distrust"
    prompt = build_prompt(sample_conversation, use_case='3', custom_prompt=custom_prompt_text)
    assert "Distrust or Suspicion" in prompt or custom_prompt_text in prompt
    assert "No. There is no message that matches the prompt in the given conversation." in prompt
    assert "[Message 1 - Name]" in prompt

def test_invalid_use_case(sample_conversation):
    with pytest.raises(ValueError, match=r"Invalid use_case: 4. Must be 1 \(Actus Reus\), 2 \(Mens Rea\), or 3 \(Custom Prompt\)."):
        build_prompt(sample_conversation, use_case='4')

@pytest.fixture
def sample_conversations():
    return [
        "[Message 1] Liam: I broke the window.",
        "[Message 2] Rachel: Let's plan the robbery."
    ]

@patch("scripts.api.pipeline.inference_runner.predict")
@patch("scripts.api.pipeline.inference_runner.Model")
def test_process_conversations_valid(mock_model_class, mock_predict, sample_conversations):
    mock_predict.return_value = "Sample Output"
    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance
    mock_model_instance.load_model.return_value = None

    result = process_conversations("MISTRAL7B", sample_conversations, usecase="1")

    assert result == ["Sample Output", "Sample Output"]
    mock_model_class.assert_called_once_with("mistral:7b-instruct")
    mock_model_instance.load_model.assert_called_once()
    assert mock_predict.call_count == 2

@patch("scripts.api.pipeline.inference_runner.predict")
@patch("scripts.api.pipeline.inference_runner.Model")
def test_process_conversations_empty(mock_model_class, mock_predict):
    result = process_conversations("MISTRAL7B", [], usecase="1")
    assert result == []
    mock_model_class.assert_called_once()
    mock_predict.assert_not_called()


@patch("scripts.api.pipeline.inference_runner.predict")
@patch("scripts.api.pipeline.inference_runner.Model")
def test_process_conversations_invalid_usecase(mock_model_class, mock_predict, sample_conversations):
    mock_model_class.return_value.load_model.return_value = None
    with pytest.raises(ValueError, match=r"Invalid use_case: 4. Must be 1 \(Actus Reus\), 2 \(Mens Rea\), or 3 \(Custom Prompt\)."):
        process_conversations("MISTRAL7B", sample_conversations, usecase="4")

@patch("scripts.api.pipeline.inference_runner.ollama.chat")
def test_predict_valid(mock_chat):
    mock_chat.return_value = {
        "message": {"content": "This is the generated output."}
    }

    output = predict("mistral:7b-instruct", "Some prompt")
    assert output == "This is the generated output."
    mock_chat.assert_called_once_with(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": "Some prompt"}]
    )

@patch("scripts.api.pipeline.inference_runner.ollama.chat")
def test_predict_invalid_model(mock_chat):
    mock_chat.side_effect = Exception("Model not found")

    output = predict("invalid-model", "Test prompt")
    assert output.startswith("Error: Model not found")