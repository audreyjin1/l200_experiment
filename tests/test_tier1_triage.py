import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from healthcare_agent.triage.tier1_deterministic import Tier1DeterministicTriage


class TestTier1DeterministicTriage(unittest.TestCase):
    """Verify deterministic, zero-latency detection of acute medical emergencies."""

    def setUp(self):
        self.triage = Tier1DeterministicTriage()

    def test_cardiovascular_emergency_detected(self):
        prompts = [
            "I have crushing chest pain and feel nauseous",
            "There is intense chest pain radiating to my left arm and jaw",
            "Severe tightness in chest with cold sweat",
            "Help, I think I am having a heart attack right now",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = self.triage.evaluate(prompt)
                self.assertTrue(result.is_red_flag, f"Failed to detect red flag in: {prompt}")
                self.assertEqual(result.matched_condition, "Cardiovascular Emergency")

    def test_stroke_fast_symptoms_detected(self):
        prompts = [
            "My mother's facial drooping started 20 minutes ago",
            "Sudden arm weakness on one side, can't lift my left arm",
            "He has sudden slurred speech and confusion",
            "This is the worst headache of my life, came out of nowhere like thunderclap",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = self.triage.evaluate(prompt)
                self.assertTrue(result.is_red_flag, f"Failed to detect stroke signs: {prompt}")
                self.assertEqual(result.matched_condition, "Neurological Emergency / Stroke")

    def test_respiratory_emergency_detected(self):
        prompts = [
            "Patient cannot breathe and is choking",
            "He is gasping for air and his lips are blue",
            "Child with severe stridor and turning cyanotic",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = self.triage.evaluate(prompt)
                self.assertTrue(result.is_red_flag, f"Failed to detect respiratory red flag: {prompt}")
                self.assertEqual(result.matched_condition, "Respiratory Emergency")

    def test_anaphylaxis_detected(self):
        prompts = [
            "I ate peanuts and my airway is swelling shut",
            "My tongue is swollen and can't breathe after bee sting",
            "Having severe allergic reaction and cannot breathe",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = self.triage.evaluate(prompt)
                self.assertTrue(result.is_red_flag, f"Failed to detect anaphylaxis: {prompt}")
                self.assertEqual(result.matched_condition, "Anaphylaxis Emergency")

    def test_severe_hemorrhage_trauma_detected(self):
        prompts = [
            "Deep laceration with spurting blood on arm",
            "Bleeding won't stop after direct pressure for 20 minutes",
            "Compound fracture with bone sticking out",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = self.triage.evaluate(prompt)
                self.assertTrue(result.is_red_flag, f"Failed to detect hemorrhage/trauma: {prompt}")
                self.assertEqual(result.matched_condition, "Severe Hemorrhage / Critical Trauma")

    def test_mental_health_crisis_detected(self):
        prompts = [
            "I am going to kill myself tonight",
            "I want to swallow all my pills to end it",
            "I have suicidal thoughts with a plan to hang myself",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = self.triage.evaluate(prompt)
                self.assertTrue(result.is_red_flag, f"Failed to detect suicide crisis: {prompt}")
                self.assertEqual(result.matched_condition, "Mental Health Crisis / Acute Self-Harm")

    def test_routine_symptoms_do_not_trigger_false_positive(self):
        routine_prompts = [
            "I have a mild sore throat and a slight cough",
            "I twisted my ankle yesterday while jogging, it hurts a little to walk",
            "What can I do for a small paper cut on my index finger?",
            "My allergies are acting up and my eyes are watery",
            "Can I get a prescription refill for metformin?",
            "What are your clinic operating hours on Sunday?",
            "How much water should I drink every day?",
        ]
        for prompt in routine_prompts:
            with self.subTest(prompt=prompt):
                result = self.triage.evaluate(prompt)
                self.assertFalse(result.is_red_flag, f"False positive triggered for routine prompt: {prompt}")


if __name__ == "__main__":
    unittest.main()
