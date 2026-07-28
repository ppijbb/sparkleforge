"""
Universal System Domain Vocabulary Translator.

Provides a translation layer to map domain-specific jargon into a 
standardized systems-theory vocabulary to facilitate cross-domain synthesis.
"""

import logging

class VocabularyTranslator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Mapping of domain-specific terms to universal system primitives
        self._registry = {
            "iot": "actuation_plane",
            "sensor": "observation_node",
            "database": "state_store",
            "api": "invocation_gateway",
            "user": "external_agent",
            "task": "workflow_unit",
            "error": "anomaly_event"
        }

    def translate(self, text: str) -> str:
        """Translates domain jargon to universal systems language."""
        words = text.split()
        translated = []
        for word in words:
            clean_word = word.lower().strip(".,!?")
            if clean_word in self._registry:
                replacement = self._registry[clean_word]
                # Preserve original punctuation if possible
                translated.append(word.lower().replace(clean_word, replacement))
            else:
                translated.append(word)
        return " ".join(translated)

    def get_universal_term(self, domain_term: str) -> str:
        return self._registry.get(domain_term.lower(), domain_term)
