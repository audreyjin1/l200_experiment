import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from healthcare_agent.data_protection.sdp_deidentifier import CloudSDPDeidentifier


class TestCloudSDPDeidentifier(unittest.TestCase):
    """Verify PHI tokenization and surrogate detokenization."""

    def setUp(self):
        self.sdp = CloudSDPDeidentifier()

    def test_direct_phi_tokenization(self):
        input_text = (
            "Patient: John Doe, DOB: 1982-04-15, MRN 482910. "
            "Contact: 555-123-4567, email: jdoe@example.com, SSN: 123-45-6789. "
            "Address: 742 Evergreen Terrace. Complains of mild headache."
        )
        result = self.sdp.deidentify(input_text)

        # Ensure direct identifiers are NOT present in tokenized text
        self.assertNotIn("John Doe", result.tokenized_text)
        self.assertNotIn("1982-04-15", result.tokenized_text)
        self.assertNotIn("482910", result.tokenized_text)
        self.assertNotIn("555-123-4567", result.tokenized_text)
        self.assertNotIn("jdoe@example.com", result.tokenized_text)
        self.assertNotIn("123-45-6789", result.tokenized_text)
        self.assertNotIn("742 Evergreen Terrace", result.tokenized_text)

        # Ensure surrogate tokens exist
        self.assertIn("[PATIENT_NAME_1]", result.tokenized_text)
        self.assertIn("[DATE_1]", result.tokenized_text)
        self.assertIn("[MEDICAL_ID_1]", result.tokenized_text)
        self.assertIn("[PHONE_1]", result.tokenized_text)
        self.assertIn("[EMAIL_1]", result.tokenized_text)
        self.assertIn("[SSN_1]", result.tokenized_text)
        self.assertIn("[ADDRESS_1]", result.tokenized_text)

        # Ensure clinical symptom context is preserved
        self.assertIn("mild headache", result.tokenized_text)
        self.assertGreaterEqual(result.tokens_count, 6)

    def test_detokenization_consistency(self):
        input_text = "My name is Sarah Connor, born 1985-02-28, MRN 99281. My knee hurts."
        deid_result = self.sdp.deidentify(input_text)

        restored_text = self.sdp.detokenize(
            deid_result.tokenized_text,
            deid_result.surrogate_mapping,
        )

        self.assertIn("Sarah Connor", restored_text)
        self.assertIn("1985-02-28", restored_text)
        self.assertIn("99281", restored_text)
        self.assertIn("My knee hurts.", restored_text)

    def test_clean_input_without_phi_remains_unchanged(self):
        clean_text = "I sprained my ankle while running on a trail. Can I use an ice pack?"
        result = self.sdp.deidentify(clean_text)

        self.assertEqual(result.tokenized_text, clean_text)
        self.assertEqual(result.tokens_count, 0)
        self.assertEqual(len(result.surrogate_mapping), 0)


if __name__ == "__main__":
    unittest.main()
