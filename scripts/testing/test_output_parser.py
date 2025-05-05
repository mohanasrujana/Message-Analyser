from scripts.api.pipeline.output_parsing import OutputParser
import pytest
import pandas as pd

@pytest.fixture
def parser():
    return OutputParser(output_dir="dummy_dir", output_file_type="csv")

def test_parse_results_yes(parser):
    output = """
    Yes. Evidence:
    Actus Reus:
    [Message 1 - John]: I hit him.
    [Message 2 - John]: He fell down.
    """

    result = parser.parse_results_grouped(output, 1, 1)
    assert result["Answer"] == "Yes"
    assert result["Evidence"]["category"] == "Actus Reus"
    assert "[Message 1 - John]: I hit him." in result["Evidence"]["message_text"]

def test_parse_results_no(parser):
    output = "No. There is no message that matches the prompt in the given conversation."
    result = parser.parse_results_grouped(output, 1, 1)

    assert result["Answer"] == "No"
    assert isinstance(result["Evidence"]["message_text"], list)
    assert "There is no message that matches the prompt in the given conversation." in result["Evidence"]["message_text"][0]

def test_clean_text_ascii_conversion(parser):
    fancy = "café résumé naïve"
    result = parser.clean_text_for_pdf(fancy)
    assert result == "cafe resume naive"

def test_save_to_csv(tmp_path):
    parser = OutputParser(output_dir=str(tmp_path), output_file_type="csv")
    output = """
    Yes. Evidence:
    Actus Reus:
    [Message 1 - John]: I hit him.
    [Message 2 - John]: He fell down.
    """

    results = [parser.parse_results_grouped(output, 1, 1)]
    
    parser.save_to_file(results)
    
    saved_file = tmp_path / "parsed_output.csv"
    assert saved_file.exists()
    content = saved_file.read_text()
    assert "Actus Reus" in content
    assert "[Message 1 - John]: I hit him." in content

def test_parse_results_malformed_message(parser):
    output = """
    Yes. Evidence:
    Actus Reus:
    [Message 1 John] I hit him.
    [Message 2 - John]: He fell down.
    """
    result = parser.parse_results_grouped(output, 1, 1)
    print(result["Evidence"]["message_text"])

    assert result["Answer"] == "Yes"
    assert result["Evidence"]["category"] == "Actus Reus"
    assert "[Message 1 John] I hit him." in result["Evidence"]["message_text"]
    assert "[Message 2 - John]: He fell down." in result["Evidence"]["message_text"]

def test_parse_results_missing_category(parser):
    output = """
    Yes. Evidence:
    [Message 1 - John]: I did it.
    """
    result = parser.parse_results_grouped(output, 1, 1)

    assert result["Answer"] == "Yes"
    assert result["Evidence"]["category"] is None
    assert "[Message 1 - John]: I did it." in result["Evidence"]["message_text"]

def test_parse_results_empty_input(parser):
    output = "\n"
    result = parser.parse_results_grouped(output, 1, 1)

    assert result["Answer"] is None
    assert result["Evidence"]["category"] is None
    assert result["Evidence"]["message_text"] is None


def test_csv_output_format_evidence(tmp_path):
    parser = OutputParser(output_dir=str(tmp_path), output_file_type="csv")
    
    raw_output = """
    Yes. Evidence:
    Mens Rea:
    [Message 1 - A]: I planned this a week ago.
    [Message 2 - B]: He knew what he was doing.
    """
    
    result = parser.parse_results_grouped(raw_output, 1, 1)
    parser.save_to_file([result])
    
    # Read back saved CSV
    saved_file = tmp_path / "parsed_output.csv"
    df = pd.read_csv(saved_file)

    # Validate format
    assert list(df.columns) == ["conversation_id", "category", "message_text"]
    assert df.iloc[0]["conversation_id"] == 1
    assert df.iloc[0]["category"] == "Mens Rea"
    assert "[Message 1 - A]" in df.iloc[0]["message_text"]
    assert "[Message 2 - B]" in df.iloc[0]["message_text"]


def test_csv_output_format_no_evidence(tmp_path):
    parser = OutputParser(output_dir=str(tmp_path), output_file_type="csv")

    raw_output = "No. There is no message that matches the prompt in the given conversation."
    result = parser.parse_results_grouped(raw_output, 1, 1)
    parser.save_to_file([result])
    
    # Read back saved CSV
    saved_file = tmp_path / "parsed_output.csv"
    df = pd.read_csv(saved_file)
    assert list(df.columns) == ["conversation_id", "category", "message_text"]
    assert df.iloc[0]["conversation_id"] == 1
    assert pd.isna(df.iloc[0]["category"])
    assert "There is no message that matches the prompt in the given conversation." in df.iloc[0]["message_text"]
    
def test_xlsx_output_format_evidence(tmp_path):
    parser = OutputParser(output_dir=str(tmp_path), output_file_type="xlsx")
    
    raw_output = """
    Yes. Evidence:
    Mens Rea:
    [Message 1 - A]: I planned this a week ago.
    [Message 2 - B]: He knew what he was doing.
    """
    
    result = parser.parse_results_grouped(raw_output, 1, 1)
    parser.save_to_file([result])
    
    # Read back saved CSV
    saved_file = tmp_path / "parsed_output.xlsx"
    df = pd.read_excel(saved_file)

    # Validate format
    assert list(df.columns) == ["conversation_id", "category", "message_text"]
    assert df.iloc[0]["conversation_id"] == 1
    assert df.iloc[0]["category"] == "Mens Rea"
    assert "[Message 1 - A]" in df.iloc[0]["message_text"]
    assert "[Message 2 - B]" in df.iloc[0]["message_text"]


def test_xlsx_output_format_no_evidence(tmp_path):
    parser = OutputParser(output_dir=str(tmp_path), output_file_type="xlsx")

    raw_output = "No. There is no message that matches the prompt in the given conversation."
    result = parser.parse_results_grouped(raw_output, 1, 1)
    parser.save_to_file([result])
    
    # Read back saved CSV
    saved_file = tmp_path / "parsed_output.xlsx"
    df = pd.read_excel(saved_file)
    assert list(df.columns) == ["conversation_id", "category", "message_text"]
    assert df.iloc[0]["conversation_id"] == 1
    assert pd.isna(df.iloc[0]["category"])
    assert "There is no message that matches the prompt in the given conversation." in df.iloc[0]["message_text"]

def test_txt_output_format_evidence(tmp_path):
    parser = OutputParser(output_dir=str(tmp_path), output_file_type="txt")

    raw_output = """
    Yes. Evidence:
    Mens Rea:
    [Message 1 - A]: I knew what I was doing.
    """
    result = parser.parse_results_grouped(raw_output, 1, 1)
    parser.save_to_file([result])

    file_path = tmp_path / "parsed_output.txt"
    assert file_path.exists()

    content = file_path.read_text()
    assert "1 | Mens Rea | [Message 1 - A]: I knew what I was doing." in content

def test_txt_output_format_no_evidence(tmp_path):
    parser = OutputParser(output_dir=str(tmp_path), output_file_type="txt")

    raw_output = "No. There is no message that matches the prompt in the given conversation."
    result = parser.parse_results_grouped(raw_output, 1, 1)
    parser.save_to_file([result])

    file_path = tmp_path / "parsed_output.txt"
    assert file_path.exists()

    content = file_path.read_text()
    assert "1 | None | There is no message that matches the prompt in the given conversation." in content

def test_pdf_output_file_created(tmp_path):
    parser = OutputParser(output_dir=str(tmp_path), output_file_type="pdf")

    raw_output = """
    Yes. Evidence:
    Actus Reus:
    [Message 1 - A]: I broke the lock.
    """
    result = parser.parse_results_grouped(raw_output, 1, 1)
    parser.save_to_file([result])

    file_path = tmp_path / "parsed_output.pdf"
    assert file_path.exists()
    assert file_path.stat().st_size > 0  # non-empty