"""
Comparison report: SHAP vs LIME for AWG Solar Prediction Model

This script generates a side-by-side comparison to help your team understand
when to use each explainability method.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def generate_comparison_report(output_dir: str):
    """Generate a comparison document between SHAP and LIME.
    
    Args:
        output_dir: Directory to save the comparison report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visual comparison chart
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Title
    fig.suptitle('SHAP vs LIME: Explainable AI Comparison for AWG Solar Prediction',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Create comparison table
    categories = [
        'Theoretical Foundation',
        'Computation Speed',
        'Accuracy',
        'Consistency',
        'Global Explanations',
        'Local Explanations',
        'Feature Interactions',
        'Ease of Understanding',
        'Best Use Case'
    ]
    
    shap_values = [
        'Game Theory\n(Shapley values)',
        'Slower\n(exact computation)',
        'Exact & precise',
        'Deterministic\n(same result every time)',
        'Excellent\n(feature importance)',
        'Excellent\n(waterfall/force plots)',
        'Captures well\n(interaction effects)',
        'Medium\n(needs some explanation)',
        'Deep model understanding,\nfeature importance analysis'
    ]
    
    lime_values = [
        'Local linear\napproximation',
        'Faster\n(sampling-based)',
        'Approximate',
        'Varies slightly\n(uses random sampling)',
        'Limited\n(aggregate local models)',
        'Excellent\n(simple coefficients)',
        'Limited\n(linear assumptions)',
        'High\n(intuitive for stakeholders)',
        'Explaining specific predictions\nto non-technical audience'
    ]
    
    # Color coding
    shap_color = '#FF9999'  # Light red
    lime_color = '#9999FF'  # Light blue
    
    # Draw table
    y_start = 0.85
    row_height = 0.08
    col_width = 0.3
    
    # Headers
    ax.text(0.15, y_start + 0.05, 'Category', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.40, y_start + 0.05, 'SHAP', fontsize=12, fontweight='bold', ha='center', 
            color='darkred')
    ax.text(0.75, y_start + 0.05, 'LIME', fontsize=12, fontweight='bold', ha='center',
            color='darkblue')
    
    # Rows
    for i, (cat, shap_val, lime_val) in enumerate(zip(categories, shap_values, lime_values)):
        y_pos = y_start - (i * row_height)
        
        # Category
        ax.text(0.15, y_pos, cat, fontsize=10, ha='center', va='center', fontweight='bold')
        
        # SHAP
        rect = Rectangle((0.25, y_pos - row_height/2.5), col_width, row_height*0.8,
                         facecolor=shap_color, edgecolor='black', alpha=0.3)
        ax.add_patch(rect)
        ax.text(0.40, y_pos, shap_val, fontsize=9, ha='center', va='center',
                wrap=True, multialignment='center')
        
        # LIME
        rect = Rectangle((0.60, y_pos - row_height/2.5), col_width, row_height*0.8,
                         facecolor=lime_color, edgecolor='black', alpha=0.3)
        ax.add_patch(rect)
        ax.text(0.75, y_pos, lime_val, fontsize=9, ha='center', va='center',
                wrap=True, multialignment='center')
    
    # Recommendation box
    y_rec = y_start - (len(categories) * row_height) - 0.05
    ax.text(0.5, y_rec, '💡 RECOMMENDATION FOR YOUR TEAM', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    recommendation = (
        "Use BOTH methods together:\n\n"
        "• SHAP: For technical analysis, understanding feature importance,\n"
        "  and identifying which weather conditions matter most\n\n"
        "• LIME: For presentations to stakeholders, explaining specific\n"
        "  predictions (e.g., 'Why was water yield low on June 15th?')"
    )
    
    ax.text(0.5, y_rec - 0.15, recommendation,
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_vs_lime_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate detailed text report
    report_path = os.path.join(output_dir, 'comparison_guide.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("SHAP vs LIME: Complete Comparison Guide for AWG Solar Prediction Model\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("WHAT IS EXPLAINABLE AI (XAI)?\n")
        f.write("-" * 100 + "\n")
        f.write("Explainable AI helps us understand WHY a machine learning model makes certain predictions.\n")
        f.write("For our AWG system, XAI answers questions like:\n")
        f.write("  • Why did the model predict low solar energy on a particular day?\n")
        f.write("  • Which weather conditions have the biggest impact on water yield?\n")
        f.write("  • Can we trust the model's predictions for operational decisions?\n\n")
        
        f.write("WHY DO WE NEED XAI?\n")
        f.write("-" * 100 + "\n")
        f.write("1. Trust: Understand and validate model behavior before deployment\n")
        f.write("2. Debugging: Identify if model learned correct patterns or biases\n")
        f.write("3. Insights: Discover which weather conditions matter most\n")
        f.write("4. Compliance: Explain predictions to stakeholders/regulators\n")
        f.write("5. Improvement: Guide feature engineering and data collection\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("SHAP (SHapley Additive exPlanations)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("What it is:\n")
        f.write("-" * 100 + "\n")
        f.write("SHAP uses game theory (Shapley values from cooperative game theory) to fairly\n")
        f.write("distribute prediction 'credit' among features. It computes the exact contribution\n")
        f.write("of each feature to every prediction.\n\n")
        
        f.write("How it works:\n")
        f.write("-" * 100 + "\n")
        f.write("1. Takes a prediction (e.g., GHI = 650 W/m²)\n")
        f.write("2. Compares to baseline (average prediction across all data)\n")
        f.write("3. Calculates how much each feature pushed prediction up or down\n")
        f.write("4. Guarantees contributions sum to total prediction difference\n\n")
        
        f.write("Strengths:\n")
        f.write("-" * 100 + "\n")
        f.write("+ Mathematically rigorous (based on Shapley values)\n")
        f.write("+ Consistent - same input always gives same explanation\n")
        f.write("+ Exact for tree-based models (Random Forest)\n")
        f.write("+ Captures feature interactions well\n")
        f.write("+ Provides both global (all predictions) and local (single prediction) explanations\n\n")
        
        f.write("Weaknesses:\n")
        f.write("-" * 100 + "\n")
        f.write("- Can be slow for large datasets\n")
        f.write("- Requires some technical knowledge to interpret\n")
        f.write("- Visualizations can be complex for non-technical audiences\n\n")
        
        f.write("Best for:\n")
        f.write("-" * 100 + "\n")
        f.write("• Technical team analysis\n")
        f.write("• Understanding overall feature importance\n")
        f.write("• Validating model learned correct patterns\n")
        f.write("• Scientific publications or technical reports\n")
        f.write("• When you need exact, reproducible explanations\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("LIME (Local Interpretable Model-agnostic Explanations)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("What it is:\n")
        f.write("-" * 100 + "\n")
        f.write("LIME explains individual predictions by fitting a simple, interpretable model\n")
        f.write("(linear regression) locally around the prediction you want to explain.\n\n")
        
        f.write("How it works:\n")
        f.write("-" * 100 + "\n")
        f.write("1. Takes a prediction you want to explain\n")
        f.write("2. Creates many similar 'synthetic' examples by perturbing features\n")
        f.write("3. Gets model predictions for these synthetic examples\n")
        f.write("4. Fits a simple linear model to approximate behavior in that local region\n")
        f.write("5. Linear model coefficients = feature importance for that prediction\n\n")
        
        f.write("Strengths:\n")
        f.write("-" * 100 + "\n")
        f.write("+ Fast - uses sampling instead of exact computation\n")
        f.write("+ Easy to understand - simple linear relationships\n")
        f.write("+ Model-agnostic - works with any black-box model\n")
        f.write("+ Great for explaining specific predictions to stakeholders\n")
        f.write("+ Intuitive visualizations\n\n")
        
        f.write("Weaknesses:\n")
        f.write("-" * 100 + "\n")
        f.write("- Approximate - uses local linear model, not exact\n")
        f.write("- Can vary between runs (uses random sampling)\n")
        f.write("- Only explains local behavior, not global patterns\n")
        f.write("- Linear assumption may miss complex interactions\n")
        f.write("- Needs careful tuning of number of samples\n\n")
        
        f.write("Best for:\n")
        f.write("-" * 100 + "\n")
        f.write("• Explaining specific predictions to non-technical stakeholders\n")
        f.write("• 'Why did the model predict X for this particular day?' questions\n")
        f.write("• Quick exploratory analysis\n")
        f.write("• Presentations and demos\n")
        f.write("• When speed is more important than exactness\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("PRACTICAL GUIDE FOR YOUR TEAM\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Use SHAP when:\n")
        f.write("-" * 100 + "\n")
        f.write("1. Analyzing which features are most important overall\n")
        f.write("2. Validating the model learned correct physical relationships\n")
        f.write("   (e.g., solar zenith angle should have strong impact)\n")
        f.write("3. Writing technical documentation or papers\n")
        f.write("4. Debugging unexpected model behavior\n")
        f.write("5. Comparing feature importance across different model versions\n\n")
        
        f.write("Use LIME when:\n")
        f.write("-" * 100 + "\n")
        f.write("1. Explaining a specific prediction to management/customers\n")
        f.write("2. Investigating anomalous predictions (outliers)\n")
        f.write("3. Creating simple, intuitive visualizations for presentations\n")
        f.write("4. Quick exploratory analysis during development\n")
        f.write("5. Demonstrating model behavior to non-technical stakeholders\n\n")
        
        f.write("Example Questions and Which Tool to Use:\n")
        f.write("-" * 100 + "\n")
        f.write("Q: 'What weather factors matter most for our AWG system?'\n")
        f.write("A: Use SHAP - provides global feature importance across all predictions\n\n")
        
        f.write("Q: 'Why did we get low water yield on June 15th?'\n")
        f.write("A: Use LIME - explains that specific prediction in simple terms\n\n")
        
        f.write("Q: 'Does the model correctly account for humidity effects?'\n")
        f.write("A: Use SHAP - can validate feature relationships match physics\n\n")
        
        f.write("Q: 'Can you explain this prediction to our investors?'\n")
        f.write("A: Use LIME - provides intuitive, easy-to-understand explanations\n\n")
        
        f.write("Q: 'Which features should we monitor most carefully in production?'\n")
        f.write("A: Use SHAP - identifies most influential features globally\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("RECOMMENDATION: Use Both!\n")
        f.write("=" * 100 + "\n")
        f.write("SHAP and LIME are complementary, not competing tools.\n\n")
        f.write("Workflow:\n")
        f.write("1. Run SHAP first for comprehensive feature importance analysis\n")
        f.write("2. Use SHAP results to validate model and identify key features\n")
        f.write("3. Run LIME for specific predictions you need to explain\n")
        f.write("4. Use SHAP for technical documentation, LIME for presentations\n\n")
        
        f.write("For your AWG model:\n")
        f.write("• SHAP helps engineers understand and improve the system\n")
        f.write("• LIME helps explain predictions to operations/business teams\n")
        f.write("• Together, they provide complete explainability coverage\n\n")
    
    print(f"\n✓ Comparison report generated!")
    print(f"✓ Visual comparison: {output_dir}/shap_vs_lime_comparison.png")
    print(f"✓ Detailed guide: {report_path}")
    
    return report_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SHAP vs LIME comparison report')
    parser.add_argument('--output-dir', default='comparison_output',
                        help='Output directory for comparison report')
    args = parser.parse_args()
    
    generate_comparison_report(args.output_dir)
