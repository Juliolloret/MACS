"""Tests for the :class:`PDFLoaderAgent`."""

import os
import shutil
import unittest
from unittest.mock import patch

from agents.pdf_loader_agent import PDFLoaderAgent
from llm_fake import FakeLLM
from utils import PyPDF2


class TestPDFLoaderAgent(unittest.TestCase):
    """Unit tests for ``PDFLoaderAgent``."""

    def setUp(self):
        """Create a temporary output directory and initialise the agent."""
        self.test_outputs_dir = "test_outputs"
        if not os.path.exists(self.test_outputs_dir):
            os.makedirs(self.test_outputs_dir)
        app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {},
        }
        self.agent = PDFLoaderAgent(
            "pdf_loader", "PDFLoaderAgent", {}, FakeLLM(app_config), app_config
        )

    def tearDown(self):
        """Remove the temporary output directory created in ``setUp``."""
        if os.path.exists(self.test_outputs_dir):
            shutil.rmtree(self.test_outputs_dir)

    def _create_dummy_pdf(self, file_path, text_content, encrypted=False, password=""):
        """Write a minimal PDF containing ``text_content`` to ``file_path``.

        The implementation uses a pre-built PDF snippet to avoid external
        dependencies such as reportlab.  If ``encrypted`` is ``True`` and
        ``PyPDF2`` is available the resulting PDF is password protected.
        """

        pdf_bytes = (
            b"%PDF-1.1\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n"
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page \n"
            b"/Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << "
            b"/F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf "
            b"100 700 Td (This is a test.) Tj ET\nendstream\nendobj\n5 0 obj\n<< /Type /Font \n"
            b"/Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n"
            b"0000000010 00000 n \n0000000053 00000 n \n0000000102 00000 n \n0000000211 00000 n \n"
            b"0000000290 00000 n \ntrailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n344\n%%EOF"
        )
        with open(file_path, "wb") as file:
            file.write(pdf_bytes)

        if encrypted:
            if not PyPDF2:
                raise RuntimeError("PyPDF2 library not available for encryption")
            reader = PyPDF2.PdfReader(file_path)
            writer = PyPDF2.PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            writer.encrypt(password)
            with open(file_path, "wb") as file:
                writer.write(file)

    def test_load_pdf_successfully(self):
        """PDF content is loaded and returned when the path is valid."""
        pdf_path = os.path.join(self.test_outputs_dir, "test.pdf")
        expected_text = "This is a test."
        self._create_dummy_pdf(pdf_path, expected_text)
        result = self.agent.execute({"pdf_path": pdf_path})
        self.assertIn("pdf_text_content", result)
        self.assertNotIn("error", result)
        self.assertEqual(result["pdf_text_content"].strip(), expected_text)

    def test_pdf_path_not_provided(self):
        """Execution fails when no ``pdf_path`` is supplied."""
        result = self.agent.execute({})
        self.assertIn("error", result)
        self.assertEqual(result["error"], "PDF path not provided.")

    def test_pdf_not_found(self):
        """An error is returned when the file path does not exist."""
        pdf_path = os.path.join(self.test_outputs_dir, "non_existent.pdf")
        result = self.agent.execute({"pdf_path": pdf_path})
        self.assertIn("error", result)
        self.assertEqual(result["error"], f"PDF file not found: {pdf_path}")

    def test_empty_pdf(self):
        """Loading an empty PDF yields an appropriate error message."""
        pdf_path = os.path.join(self.test_outputs_dir, "empty.pdf")
        with open(pdf_path, "w", encoding="utf-8") as file:
            file.write("")
        result = self.agent.execute({"pdf_path": pdf_path})
        self.assertIn("error", result)
        self.assertEqual(result["error"], f"PDF file is empty: {pdf_path}")

    def test_encrypted_pdf(self):
        """Encrypted PDFs that cannot be decrypted surface an error."""
        pdf_path = os.path.join(self.test_outputs_dir, "encrypted.pdf")
        self._create_dummy_pdf(pdf_path, "This is a test.", encrypted=True, password="test")
        with patch.object(PyPDF2.PdfReader, "decrypt", return_value=0):
            result = self.agent.execute({"pdf_path": pdf_path})
            self.assertIn("error", result)
            self.assertEqual(
                result["error"], f"Failed to decrypt PDF: {os.path.basename(pdf_path)}."
            )
