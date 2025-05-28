"""
Hearing Loss Module - __init__.py
"""

from .hasqi import hasqi_v2
from .fig6 import fig6_prescriptive_formula, apply_fig6_compensation, apply_fig6_to_spectrogram

__all__ = [
    'hasqi_v2',
    'fig6_prescriptive_formula',
    'apply_fig6_compensation',
    'apply_fig6_to_spectrogram'
]