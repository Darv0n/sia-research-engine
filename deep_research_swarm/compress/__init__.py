"""Knowledge compression layer (V10/Tensegrity).

Modules:
  cluster   — passage clustering by embedding similarity or heading fallback
  artifact  — KnowledgeArtifact builder from JudgmentContext + passages
  grounding — 3-tier claim verification cascade (embedding > Jaccard)
"""
