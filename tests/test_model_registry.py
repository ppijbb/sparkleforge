import unittest
from src.core.llm_manager.model_registry import ModelRegistryMixin

class _Registry(ModelRegistryMixin):
    def __init__(self):
        super().__init__()
        self.models = {}

class TestModelRegistryCostAndTemperature(unittest.TestCase):
    def test_sol_luna_cost_ratio_and_temperature(self):
        registry = _Registry()
        registry._load_openai_models()
        
        self.assertIn("gpt-6-luna", registry.models)
        self.assertIn("gpt-6-sol", registry.models)
        
        luna = registry.models["gpt-6-luna"]
        sol = registry.models["gpt-6-sol"]
        
        self.assertEqual(sol.cost_per_token, luna.cost_per_token * 100)
        self.assertIsNotNone(luna.temperature)
        self.assertIsNotNone(sol.temperature)

if __name__ == "__main__":
    unittest.main()
