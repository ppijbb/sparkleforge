# Changelog

## [Unreleased]

### Changes
- **OpenRouter Model Registry / Pricing**:
  - Updated `gpt-5-nano` pricing explicitly to `0.00005` (matching its cost reduction migration note).
  - Added explicit `temperature: 0.1` settings to new GPT-6 family models (`gpt-6-luna`, `gpt-6-sol`, `gpt-6.1-sol`) for schema consistency.
  - Strengthened cost ratio tests and test double implementations for `ModelRegistryMixin`.
