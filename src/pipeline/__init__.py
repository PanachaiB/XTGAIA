"""XTGAIA end-to-end pipeline: Role 1 (graph/risk) -> Role 2 (TGN/XAI) -> Role 3 (DQN/SOAR)."""
from src.pipeline.orchestrator import XTGAIAPipeline, load_pipeline_config

__all__ = ["XTGAIAPipeline", "load_pipeline_config"]
