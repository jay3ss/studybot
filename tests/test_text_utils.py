import unittest

from app.text.utils import get_file_extension


class TestTextUtils(unittest.TestCase):

    def test_get_file_extension_simple_filenames(self):
        correct_filenames = [
            ("file.srt", ".srt"),
            ("file.txt", ".txt"),
            ("file.docx", ".docx"),
            ("file.pptx", ".pptx"),
            ("file.pdf", ".pdf"),
        ]
        for filename, correct_ext in correct_filenames:
            extension = get_file_extension(filename)
            self.assertEqual(correct_ext, extension)

    def test_get_file_extension_complex_filenames(self):
        correct_filenames = [
            ("file.blah.srt", ".srt"),
            ("file.blah.txt", ".txt"),
            ("file.blah.docx", ".docx"),
            ("file.blah.pptx", ".pptx"),
            ("file.blah.pdf", ".pdf"),
        ]
        for filename, correct_ext in correct_filenames:
            extension = get_file_extension(filename)
            self.assertEqual(correct_ext, extension)

    def test_get_file_extension_edge_cases(self):
        test_cases = [
            ("file", ""),  # No extension
            (".hiddenfile", ""),  # Hidden file without an extension
            ("folder.with.dots/file.pdf", ".pdf"),  # Dots in folder names
            ("", ""),  # Empty string
            ("folder/file.txt", ".txt"),  # Path with separators
            ("folder\\file.txt", ".txt"),  # Windows-style path
            ("my_file@2025!.txt", ".txt"),  # Special characters in name
            ("file.PDF", ".PDF"),  # Uppercase extension
        ]
        for filename, correct_ext in test_cases:
            extension = get_file_extension(filename)
            self.assertEqual(correct_ext, extension)


if __name__ == "__main__":
    unittest.main()
