"""
Master script to generate complete explainability report for AWG model.

Runs both SHAP and LIME analyses plus comparison guide.
Perfect for presenting to your team this week!
"""

import os
import argparse
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='Generate complete XAI report for AWG RF model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (quick analysis)
  python generate_xai_report.py --model model/solar_predictor_model.joblib --csv usaWithWeather.csv

  # Full analysis with more samples (slower but more detailed)
  python generate_xai_report.py --model model/solar_predictor_model.joblib --csv usaWithWeather.csv --detailed

  # Custom output directory
  python generate_xai_report.py --model model/solar_predictor_model.joblib --csv usaWithWeather.csv --output my_xai_report
        """
    )
    
    parser.add_argument('--model', required=True, help='Path to trained RF model (joblib)')
    parser.add_argument('--csv', required=True, help='Path to training/test CSV data')
    parser.add_argument('--output', default='xai_report', help='Base output directory (default: xai_report)')
    parser.add_argument('--detailed', action='store_true', 
                        help='Run detailed analysis (more samples, slower)')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis')
    parser.add_argument('--skip-lime', action='store_true', help='Skip LIME analysis')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip comparison guide')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f"{args.output}_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    
    print("=" * 100)
    print("EXPLAINABLE AI REPORT GENERATOR - AWG Solar Prediction Model")
    print("=" * 100)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.csv}")
    print(f"Output: {base_dir}")
    print(f"Mode: {'DETAILED' if args.detailed else 'QUICK'}")
    print()
    
    # Set analysis parameters based on mode
    if args.detailed:
        shap_samples = 2000
        lime_samples = 5000
        lime_examples = 15
        shap_examples = 10
    else:
        shap_samples = 1000
        lime_samples = 2000
        lime_examples = 10
        shap_examples = 5
    
    # Run SHAP analysis
    if not args.skip_shap:
        print("\n" + "=" * 100)
        print("1/3 - Running SHAP Analysis")
        print("=" * 100)
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from explainability.explain_rf_shap import generate_shap_report
            shap_dir = os.path.join(base_dir, 'shap_analysis')
            generate_shap_report(
                args.model, 
                args.csv, 
                shap_dir,
                n_samples=shap_samples,
                n_examples=shap_examples
            )
            print(f"✓ SHAP analysis complete! Results in: {shap_dir}")
        except Exception as e:
            print(f"✗ SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n1/3 - Skipping SHAP analysis")
    
    # Run LIME analysis
    if not args.skip_lime:
        print("\n" + "=" * 100)
        print("2/3 - Running LIME Analysis")
        print("=" * 100)
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from explainability.explain_rf_lime import generate_lime_report
            lime_dir = os.path.join(base_dir, 'lime_analysis')
            generate_lime_report(
                args.model,
                args.csv,
                lime_dir,
                n_training_samples=lime_samples,
                n_examples=lime_examples,
                n_features=7
            )
            print(f"✓ LIME analysis complete! Results in: {lime_dir}")
        except Exception as e:
            print(f"✗ LIME analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n2/3 - Skipping LIME analysis")
    
    # Generate comparison guide
    if not args.skip_comparison:
        print("\n" + "=" * 100)
        print("3/3 - Generating Comparison Guide")
        print("=" * 100)
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from explainability.explain_comparison import generate_comparison_report
            comparison_dir = os.path.join(base_dir, 'comparison')
            generate_comparison_report(comparison_dir)
            print(f"✓ Comparison guide complete! Results in: {comparison_dir}")
        except Exception as e:
            print(f"✗ Comparison guide failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n3/3 - Skipping comparison guide")
    
    # Generate index/summary file
    print("\n" + "=" * 100)
    print("Generating Summary Report")
    print("=" * 100)
    
    summary_path = os.path.join(base_dir, 'README.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Explainable AI Report - AWG Solar Prediction Model\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model:** `{args.model}`\n\n")
        f.write(f"**Data:** `{args.csv}`\n\n")
        f.write(f"**Analysis Mode:** {'Detailed' if args.detailed else 'Quick'}\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 What's in This Report\n\n")
        
        if not args.skip_shap:
            f.write("### 🎯 SHAP Analysis (`shap_analysis/`)\n\n")
            f.write("Game theory-based explanations showing exact feature contributions.\n\n")
            f.write("**Key Files:**\n")
            f.write("- `shap_summary.txt` - Feature importance and interpretation guide\n")
            f.write("- `1_shap_feature_importance.png` - Which features matter most\n")
            f.write("- `2_shap_summary_beeswarm.png` - Feature impact distribution\n")
            f.write("- `3_*_dependence_*.png` - How feature values affect predictions\n")
            f.write("- `4_*_waterfall_*.png` - Individual prediction breakdowns\n")
            f.write("- `5_shap_force_plot.png` - Visual prediction decomposition\n\n")
        
        if not args.skip_lime:
            f.write("### 🔍 LIME Analysis (`lime_analysis/`)\n\n")
            f.write("Simple, interpretable local explanations for specific predictions.\n\n")
            f.write("**Key Files:**\n")
            f.write("- `lime_summary.txt` - Detailed explanations for all examples\n")
            f.write("- `lime_example_*.png` - Individual prediction explanations\n")
            f.write("- `lime_overview.png` - Summary of all analyzed predictions\n\n")
        
        if not args.skip_comparison:
            f.write("### 📖 Comparison Guide (`comparison/`)\n\n")
            f.write("Side-by-side comparison of SHAP vs LIME with usage recommendations.\n\n")
            f.write("**Key Files:**\n")
            f.write("- `comparison_guide.txt` - Complete guide on when to use each method\n")
            f.write("- `shap_vs_lime_comparison.png` - Visual comparison chart\n\n")
        
        f.write("---\n\n")
        f.write("## 🚀 Quick Start for Team Presentation\n\n")
        f.write("### For Technical Audience:\n")
        f.write("1. Start with `shap_analysis/1_shap_feature_importance.png`\n")
        f.write("2. Show `shap_analysis/2_shap_summary_beeswarm.png`\n")
        f.write("3. Explain with `shap_analysis/shap_summary.txt`\n\n")
        
        f.write("### For Non-Technical Audience:\n")
        f.write("1. Start with `comparison/shap_vs_lime_comparison.png`\n")
        f.write("2. Show `lime_analysis/lime_overview.png`\n")
        f.write("3. Pick 2-3 examples from `lime_analysis/lime_example_*.png`\n\n")
        
        f.write("### For Management:\n")
        f.write("1. Show `comparison/shap_vs_lime_comparison.png`\n")
        f.write("2. Key message: 'We can explain every prediction the model makes'\n")
        f.write("3. Demonstrate with one LIME example\n\n")
        
        f.write("---\n\n")
        f.write("## 💡 Key Insights to Highlight\n\n")
        f.write("1. **Model Transparency:** Every prediction can be explained\n")
        f.write("2. **Feature Importance:** We know which weather factors matter most\n")
        f.write("3. **Validation:** Model learned correct physical relationships\n")
        f.write("4. **Trust:** Explanations help validate model for deployment\n")
        f.write("5. **Debugging:** Can identify and fix unexpected behavior\n\n")
        
        f.write("---\n\n")
        f.write("## 🎓 Understanding the Methods\n\n")
        f.write("**SHAP (SHapley Additive exPlanations)**\n")
        f.write("- Based on game theory\n")
        f.write("- Exact, mathematically rigorous\n")
        f.write("- Best for: Understanding overall model behavior\n\n")
        
        f.write("**LIME (Local Interpretable Model-agnostic Explanations)**\n")
        f.write("- Local linear approximation\n")
        f.write("- Fast, intuitive explanations\n")
        f.write("- Best for: Explaining specific predictions to stakeholders\n\n")
        
        f.write("---\n\n")
        f.write("## 📞 Questions?\n\n")
        f.write("Refer to:\n")
        f.write("- `comparison/comparison_guide.txt` for detailed methodology\n")
        f.write("- `shap_analysis/shap_summary.txt` for SHAP interpretation\n")
        f.write("- `lime_analysis/lime_summary.txt` for LIME interpretation\n")
    
    print(f"\n✓ Summary report created: {summary_path}")
    
    print("\n" + "=" * 100)
    print("🎉 EXPLAINABLE AI REPORT COMPLETE!")
    print("=" * 100)
    print(f"\n📁 All results saved to: {base_dir}")
    print(f"📖 Start with: {summary_path}")
    print("\n💡 Tip: Open README.md for a guide on presenting to your team!")
    print()


if __name__ == '__main__':
    main()
