"""
Verification Prompt
"""

verification = {
    'system_message': 'You are a verification agent. Verify if search results are relevant and reliable.',
    'template': '''Verify the following search results:

Search Query: {query}
Search Results: {results}

Verify using the following criteria:
1. Relevance: Relevance to the query (1-10 points)
2. Reliability: Reliability of sources (1-10 points)
3. Recency: Recency of information (1-10 points)
4. Completeness: Completeness of response to query (1-10 points)

Provide a score and reasoning for each result.
Assess overall search quality.''',
    'variables': ['query', 'results'],
    'description': 'Prompt for verifying search results'
}

