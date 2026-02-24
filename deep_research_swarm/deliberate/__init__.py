"""Deliberation panel — 4-judge structured evidence evaluation (V10/Tensegrity).

Judges:
  authority_judge   — source credibility via scoring/authority.py
  grounding_judge   — claim verification via scoring/grounding.py + claim_graph.py
  contradiction_judge — conflict detection via agents/contradiction.py
  coverage_judge    — facet coverage via scoring/diversity.py + gap analysis
"""
