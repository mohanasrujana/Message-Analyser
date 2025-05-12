import pytest
import pandas as pd
from io import BytesIO
from types import SimpleNamespace
from pathlib import Path
from scripts.api.pipeline.MessageAnalyserInputsHandler import MessageAnalyserInputsHandler
from unittest.mock import MagicMock, patch

def test_csv_conversion(monkeypatch, tmp_path):
    csv_content = b"message,timestamp\nHello there,123\nHow are you?,124\n,125\nI'm fine!,126"

    mock_file_input = SimpleNamespace(
        file=BytesIO(csv_content),
        filename="dummy.csv"
    )

    def fake_save_input_to_tempfile(self, file_input):
        path = tmp_path / "test.csv"
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save_input_to_tempfile)

    handler = MessageAnalyserInputsHandler(inputs={"input_file": mock_file_input})
    conversations = handler.load_conversations()

    assert conversations == ["Hello there", "How are you?", "I'm fine!"]

def test_multiple_csv_inputs(monkeypatch, tmp_path):
    csv_1 = b"message\nHello\nHow are you?"
    csv_2 = b"message\nFine\nThanks"

    file_input_1 = SimpleNamespace(file=BytesIO(csv_1), filename="file1.csv")
    file_input_2 = SimpleNamespace(file=BytesIO(csv_2), filename="file2.csv")

    def fake_save_input_to_tempfile(self, file_input):
        name = file_input.filename
        path = tmp_path / name
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save_input_to_tempfile)

    handler = MessageAnalyserInputsHandler(inputs={"input_files": SimpleNamespace(files=[file_input_1, file_input_2])})
    conversations = handler.load_conversations()

    assert conversations == ["Hello", "How are you?", "Fine", "Thanks"]

def test_csv_real_file_first_two_convs():
    test_file_path = Path("scripts/testing/data/test_conv1.csv")  # Adjust path if needed
    file_input = SimpleNamespace(path=test_file_path, filename=test_file_path.name)

    handler = MessageAnalyserInputsHandler(inputs={"input_file": file_input})
    conversations = handler.load_conversations()

    first_block = conversations[0]
    lines = [line.strip() for line in first_block.split("\n")]

    assert lines[0] == "[Message 1 - Ben]: I can't believe we pulled it off, Rach. The poison was perfectly lethal and undetectable."
    assert lines[-1] == "[Message 45 - Ben]: Excellent. Our freedom awaits us, Rach. Let's make our escape together."


def test_xlsx_conversion(monkeypatch, tmp_path):
    df = pd.DataFrame({
        "message": ["Hello", "How are you?", None, "Fine, thanks"]
    })


    xlsx_bytes = BytesIO()
    with pd.ExcelWriter(xlsx_bytes, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    xlsx_bytes.seek(0)

    file_input = SimpleNamespace(file=xlsx_bytes, filename="test.xlsx")

    def fake_save_input_to_tempfile(self, file_input):
        path = tmp_path / file_input.filename
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save_input_to_tempfile)

    handler = MessageAnalyserInputsHandler(inputs={"input_file": file_input})
    conversations = handler.load_conversations()

    assert conversations == ["Hello", "How are you?", "Fine, thanks"]

def test_multiple_xlsx_inputs(monkeypatch, tmp_path):
    df1 = pd.DataFrame({"message": ["Hello", "How are you?"]})
    df2 = pd.DataFrame({"message": ["Fine", "Thanks"]})

    def to_xlsx_bytes(df):
        xlsx = BytesIO()
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        xlsx.seek(0)
        return xlsx

    file_input_1 = SimpleNamespace(file=to_xlsx_bytes(df1), filename="file1.xlsx")
    file_input_2 = SimpleNamespace(file=to_xlsx_bytes(df2), filename="file2.xlsx")

    def fake_save_input_to_tempfile(self, file_input):
        name = file_input.filename
        path = tmp_path / name
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save_input_to_tempfile)

    handler = MessageAnalyserInputsHandler(inputs={"input_files": SimpleNamespace(files=[file_input_1, file_input_2])})
    conversations = handler.load_conversations()

    assert conversations == ["Hello", "How are you?", "Fine", "Thanks"]


def test_xlsx_real_file_first_two_convs():
    test_file_path = Path("scripts/testing/data/test_conv1.xlsx")
    file_input = SimpleNamespace(path=test_file_path, filename=test_file_path.name)

    handler = MessageAnalyserInputsHandler(inputs={"input_file": file_input})
    conversations = handler.load_conversations()

    first_block = conversations[0]
    lines = [line.strip() for line in first_block.split("\n")]

    assert lines[0] == "[Message 1 - Ben]: I can't believe we pulled it off, Rach. The poison was perfectly lethal and undetectable."
    assert lines[-1] == "[Message 45 - Ben]: Excellent. Our freedom awaits us, Rach. Let's make our escape together."

def test_txt_conversion(monkeypatch, tmp_path):
    # Simulated conversation with separators
    txt_content = b"""[Message 1 - Ben]: Hello, Rachel.
[Message 2 - Rachel]: Hi, Ben!
---
[Message 3 - Ben]: How are you?
[Message 4 - Rachel]: Doing well, thanks!"""

    # Simulated FileInput with file-like object
    mock_file_input = SimpleNamespace(file=BytesIO(txt_content), filename="dummy.txt")

    def fake_save_input_to_tempfile(self, file_input):
        path = tmp_path / "test.txt"
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save_input_to_tempfile)

    handler = MessageAnalyserInputsHandler(inputs={"input_file": mock_file_input})
    conversations = handler.load_conversations()

    assert len(conversations) == 2
    assert conversations[0].startswith("[Message 1 - Ben]: Hello, Rachel.")
    assert conversations[1].startswith("[Message 3 - Ben]: How are you?")


def test_multiple_txt_inputs(monkeypatch, tmp_path):
    txt_1 = b"""[Message 1 - A]: Hello
[Message 2 - B]: Hi!
---
[Message 3 - A]: You good?
"""
    txt_2 = b"""[Message 4 - B]: Yep!
[Message 5 - A]: Cool
---
[Message 6 - B]: Later!
"""

    file_input_1 = SimpleNamespace(file=BytesIO(txt_1), filename="file1.txt")
    file_input_2 = SimpleNamespace(file=BytesIO(txt_2), filename="file2.txt")

    def fake_save_input_to_tempfile(self, file_input):
        name = file_input.filename
        path = tmp_path / name
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save_input_to_tempfile)

    handler = MessageAnalyserInputsHandler(inputs={"input_files": SimpleNamespace(files=[file_input_1, file_input_2])})
    conversations = handler.load_conversations()

    assert len(conversations) == 4
    assert conversations[0].startswith("[Message 1 - A]: Hello")
    assert conversations[-1].startswith("[Message 6 - B]: Later!")


def test_txt_real_file_first_two_convs():
    test_file_path = Path("scripts/testing/data/test_conv1.txt")  # Adjust if needed
    file_input = SimpleNamespace(path=test_file_path, filename=test_file_path.name)

    handler = MessageAnalyserInputsHandler(inputs={"input_file": file_input})
    conversations = handler.load_conversations()

    first_block = conversations[0]
    lines = [line.strip() for line in first_block.split("\n")]

    assert lines[0] == "[Message 1 - Ben]: I can't believe we pulled it off, Rach. The poison was perfectly lethal and undetectable."
    assert lines[-1] == "[Message 45 - Ben]: Excellent. Our freedom awaits us, Rach. Let's make our escape together."

def test_pdf_conversion(monkeypatch, tmp_path):
    # Fake PDF content that mimics extracted pages
    fake_pdf_pages = [
        MagicMock(extract_text=MagicMock(return_value="[Message 1 - A]: Hello\n[Message 2 - B]: Hi")),
        MagicMock(extract_text=MagicMock(return_value="[Message 3 - A]: How are you?\n[Message 4 - B]: Good"))
    ]

    # Mock pdfplumber.open to return a context manager with .pages
    mock_pdfplumber = MagicMock()
    mock_pdfplumber.__enter__.return_value.pages = fake_pdf_pages

    # Simulated PDF file input
    fake_pdf_data = BytesIO(b"%PDF-1.4 fake content")
    file_input = SimpleNamespace(file=fake_pdf_data, filename="dummy.pdf")

    # Save the fake content to a real temporary file so our handler reads it
    def fake_save_input_to_tempfile(self, file_input):
        path = tmp_path / file_input.filename
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save_input_to_tempfile)

    with patch("scripts.api.pipeline.MessageAnalyserInputsHandler.pdfplumber.open", return_value=mock_pdfplumber):
        handler = MessageAnalyserInputsHandler(inputs={"input_file": file_input})
        conversations = handler.load_conversations()

        # One full conversation block expected
        assert len(conversations) == 1
        lines = conversations[0].split("\n")
        assert lines[0] == "[Message 1 - A]: Hello"
        assert lines[-1] == "[Message 4 - B]: Good"


def test_multiple_pdf_inputs(monkeypatch, tmp_path):
    # Fake PDF page content (for both files)
    fake_pdf_pages_1 = [MagicMock(extract_text=MagicMock(return_value="[Message 1 - A]: Hi\n[Message 2 - B]: Hello"))]
    fake_pdf_pages_2 = [MagicMock(extract_text=MagicMock(return_value="[Message 3 - A]: Later\n[Message 4 - B]: Bye"))]

    def fake_save_input_to_tempfile(self, file_input):
        path = tmp_path / file_input.filename
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save_input_to_tempfile)

    file_input_1 = SimpleNamespace(file=BytesIO(b"%PDF fake 1"), filename="file1.pdf")
    file_input_2 = SimpleNamespace(file=BytesIO(b"%PDF fake 2"), filename="file2.pdf")

    with patch("scripts.api.pipeline.MessageAnalyserInputsHandler.pdfplumber.open") as mock_open:
        mock_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=MagicMock(pages=fake_pdf_pages_1))),
            MagicMock(__enter__=MagicMock(return_value=MagicMock(pages=fake_pdf_pages_2))),
        ]

        handler = MessageAnalyserInputsHandler(inputs={"input_files": SimpleNamespace(files=[file_input_1, file_input_2])})
        conversations = handler.load_conversations()

        assert len(conversations) == 2
        assert conversations[0].startswith("[Message 1 - A]: Hi")
        assert conversations[1].startswith("[Message 3 - A]: Later")

def test_unsupported_file_type(monkeypatch, tmp_path):
    file_input = SimpleNamespace(file=BytesIO(b"irrelevant"), filename="data.docx")

    def fake_save(self, file_input):
        path = tmp_path / file_input.filename
        path.write_bytes(file_input.file.read())
        return path

    monkeypatch.setattr(MessageAnalyserInputsHandler, "_save_input_to_tempfile", fake_save)

    handler = MessageAnalyserInputsHandler(inputs={"input_file": file_input})

    with pytest.raises(ValueError, match="Unsupported file type"):
        handler.load_conversations()


def test_missing_input_keys():
    with pytest.raises(KeyError, match="Expected 'input_files' or 'input_file' in inputs"):
        MessageAnalyserInputsHandler(inputs={"other_key": "something"})

def test_invalid_file_input(monkeypatch):
    broken_input = SimpleNamespace(filename="bad.csv")  # No .file, no .path

    with patch("scripts.api.pipeline.MessageAnalyserInputsHandler.request", None):
        handler = MessageAnalyserInputsHandler(inputs={"input_file": broken_input})

        with pytest.raises(ValueError, match="Cannot extract file content"):
            handler.load_conversations()

