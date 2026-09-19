"""Thin replacement for the official drone-flyby example.py.

Keep official api.py/dtos.py/utils.py/local_evaluator.py unchanged. The official api.py
imports `predict` from this module, which delegates to the shared detector framework.
"""
from drone_detector.challenge.solution import predict

__all__ = ["predict"]
