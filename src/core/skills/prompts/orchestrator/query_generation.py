"""
Search Query Generation Prompt
"""

query_generation = {
    'system_message': 'You are a research query generator. Generate specific search queries based on research plans.',
    'template': '''Research Plan:
{plan}

Original Query: {query}

Generate 3-5 specific search queries based on the above research plan.
Each query should address different perspectives or aspects.
Response Format: Write only one search query per line. No numbers or symbols, just the queries.''',
    'variables': ['plan', 'query'],
    'description': 'Prompt for generating search queries based on research plan'
}

