"""Explainable AI module for AWG solar prediction model."""

from .explain_rf_shap import explain_with_shap, generate_shap_report
from .explain_rf_lime import explain_with_lime, generate_lime_report
from .explain_comparison import generate_comparison_report

__all__ = [
    'explain_with_shap',
    'generate_shap_report',
    'explain_with_lime',
    'generate_lime_report',
    'generate_comparison_report'
]
