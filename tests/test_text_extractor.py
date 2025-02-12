import unittest
from unittest.mock import MagicMock, mock_open, patch

from app.text.loaders import DocumentLoader


class TestDocumentLoader(unittest.TestCase):

    @patch("text.utils.get_file_extension")
    @patch("app.text.loaders.PdfReader")
    def test_extract_pdf_text(self, mock_pdf_reader, mock_get_ext):
        mock_get_ext.return_value = ".pdf"
        mock_file = mock_open()

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Sample PDF text"

        mock_pdf_reader.return_value.pages = [fake_page]

        with patch("builtins.open", mock_file, create=True):
            result = DocumentLoader.extract_text("dummy.pdf")
            self.assertEqual(result, "Sample PDF text")

    @patch("text.utils.get_file_extension")
    @patch("app.text.loaders.docx.Document")
    def test_extract_docx_text(self, mock_docx_doc, mock_get_ext):
        mock_get_ext.return_value = ".docx"
        mock_file = mock_open()

        mock_doc = MagicMock()
        mock_doc.paragraphs = [
            MagicMock(text="Paragraph 1"),
            MagicMock(text="Paragraph 2"),
        ]
        mock_docx_doc.return_value = mock_doc

        with patch("builtins.open", mock_file, create=True):
            result = DocumentLoader.extract_text("dummy.docx")
            self.assertEqual(result, "Paragraph 1\nParagraph 2")

    @patch("text.utils.get_file_extension")
    @patch("app.text.loaders.srt.parse")
    def test_extract_srt_text(self, mock_srt_parse, mock_get_ext):
        mock_get_ext.return_value = ".srt"
        mock_file = mock_open()

        subtitle1 = MagicMock(content="Subtitle line 1")
        subtitle2 = MagicMock(content="Subtitle line 2")
        mock_srt_parse.return_value = [subtitle1, subtitle2]

        with patch("builtins.open", mock_file, create=True):
            result = DocumentLoader.extract_text("dummy.srt")
            self.assertEqual(result, "Subtitle line 1\nSubtitle line 2")

    @patch("app.text.utils.get_file_extension")
    def test_extract_txt_text(self, mock_get_ext):
        mock_get_ext.return_value = ".txt"
        mock_file = mock_open(read_data="Sample text file content")

        with patch("builtins.open", mock_file, create=True):
            result = DocumentLoader.extract_text("dummy.txt")
            self.assertEqual(result, "Sample text file content")

    @patch("app.text.utils.get_file_extension")
    def test_unsupported_extension(self, mock_get_ext):
        mock_get_ext.return_value = ".xml"

        with self.assertRaises(ValueError) as context:
            DocumentLoader.extract_text("dummy.xml")

        self.assertEqual(str(context.exception), "Unsupported file extension: .xml")


if __name__ == "__main__":
    unittest.main()
