import unittest
from src.core.llm_manager.model_registry import ModelRegistryMixin

class _Registry(ModelRegistryMixin):
    def __init__(self):
        super().__init__()
        self.models = {}

class TestModelRegistryCostAndSchema(unittest.TestCase):
    def test_gpt6_cost_ratio_and_temperature(self):
        registry = _Registry()
        registry._load_openai_models()

        luna = registry.models.get("gpt-6-luna")
        sol = registry.models.get("gpt-6-sol")

        self.assertIsNotNone(luna)
        self.assertIsNotNone(sol)
        # Verify exact 100x cost ratio per documentation/issue specs
        self.assertEqual(sol.cost_per_token, luna.cost_per_token * 100)
        self.assertGreaterEqual(sol.cost_per_token, luna.cost_per_token * 100)
        self.assertEqual(luna.temperature, 0.1)

if __name__ == "__main__":
    unittest.main()
