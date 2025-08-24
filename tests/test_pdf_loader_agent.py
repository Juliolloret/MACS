import os
import sys
import unittest
import shutil
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.pdf_loader_agent import PDFLoaderAgent
from utils import PyPDF2
from llm_fake import FakeLLM

class TestPDFLoaderAgent(unittest.TestCase):
    def setUp(self):
        self.test_outputs_dir = "test_outputs"
        if not os.path.exists(self.test_outputs_dir):
            os.makedirs(self.test_outputs_dir)
        app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {}
        }
        self.agent = PDFLoaderAgent("pdf_loader", "PDFLoaderAgent", {}, FakeLLM(app_config), app_config)

    def tearDown(self):
        if os.path.exists(self.test_outputs_dir):
            shutil.rmtree(self.test_outputs_dir)

    def _create_dummy_pdf(self, file_path, text_content, encrypted=False, password=""):
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from PyPDF2 import PdfWriter, PdfReader

        writer = PdfWriter()
        reader = PdfReader(file_path) if os.path.exists(file_path) else None

        # Create a new PDF with reportlab and add it to the writer
        c = canvas.Canvas(file_path, pagesize=letter)
        c.drawString(100, 750, text_content)
        c.save()

        # Now, read the newly created PDF and add its pages to the writer
        new_pdf_reader = PdfReader(file_path)
        for page in new_pdf_reader.pages:
            writer.add_page(page)

        if encrypted:
            writer.encrypt(password)

        with open(file_path, "wb") as f:
            writer.write(f)

    def test_load_pdf_successfully(self):
        pdf_path = os.path.join(self.test_outputs_dir, "test.pdf")
        expected_text = "This is a test."
        self._create_dummy_pdf(pdf_path, expected_text)
        result = self.agent.execute({"pdf_path": pdf_path})
        self.assertIn("pdf_text_content", result)
        self.assertNotIn("error", result)
        self.assertEqual(result["pdf_text_content"].strip(), expected_text)

    def test_pdf_path_not_provided(self):
        result = self.agent.execute({})
        self.assertIn("error", result)
        self.assertEqual(result["error"], "PDF path not provided.")

    def test_pdf_not_found(self):
        pdf_path = os.path.join(self.test_outputs_dir, "non_existent.pdf")
        result = self.agent.execute({"pdf_path": pdf_path})
        self.assertIn("error", result)
        self.assertEqual(result["error"], f"PDF file not found: {pdf_path}")

    def test_empty_pdf(self):
        pdf_path = os.path.join(self.test_outputs_dir, "empty.pdf")
        with open(pdf_path, "w") as f:
            f.write("")
        result = self.agent.execute({"pdf_path": pdf_path})
        self.assertIn("error", result)
        self.assertEqual(result["error"], f"PDF file is empty: {pdf_path}")

    def test_encrypted_pdf(self):
        pdf_path = os.path.join(self.test_outputs_dir, "encrypted.pdf")
        self._create_dummy_pdf(pdf_path, "This is a test.", encrypted=True, password="test")
        # Mocking decrypt to simulate failure
        with patch.object(PyPDF2.PdfReader, "decrypt", return_value=0):
            result = self.agent.execute({"pdf_path": pdf_path})
            self.assertIn("error", result)
            self.assertEqual(result["error"], f"Failed to decrypt PDF: {os.path.basename(pdf_path)}.")
