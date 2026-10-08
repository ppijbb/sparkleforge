import unittest
from src.core.llm_manager.model_registry import ModelRegistryMixin

class _Registry(ModelRegistryMixin):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.model_clients = {}

class TestModelRegistryCostRatioAndTestDouble(unittest.TestCase):
    def test_sol_luna_cost_ratio(self):
        registry = _Registry()
        registry._load_openai_models()
        
        sol = registry.models.get("gpt-6-sol")
        luna = registry.models.get("gpt-6-luna")
        
        self.assertIsNotNone(sol)
        self.assertIsNotNone(luna)
        
        # Assert exact 100x ratio as documented in comments and required by issue specifications
        self.assertEqual(sol.cost_per_token, luna.cost_per_token * 100)

    def test_gpt_5_nano_explicit_cost(self):
        registry = _Registry()
        registry._load_openai_models()
        nano = registry.models.get("gpt-5-nano")
        self.assertIsNotNone(nano)
        self.assertEqual(nano.cost_per_token, 0.00005)

