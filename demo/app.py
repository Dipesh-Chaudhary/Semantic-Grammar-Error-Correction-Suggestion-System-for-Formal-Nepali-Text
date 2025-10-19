"""
Simple Gradio Web Demo for Nepali GEC
Launch with: python demo/app.py
"""
import gradio as gr
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.gec_pipeline import NepaliGECPipeline
from models.semantic_validator import SemanticValidator


class NepaliGECDemo:
    """Web demo for Nepali GEC"""
    
    def __init__(self, model_path: str):
        """Initialize demo with model"""
        print("Loading model...")
        self.pipeline = NepaliGECPipeline(
            model_path=model_path,
            use_semantic_validation=True
        )
        print("Model loaded successfully!")
    
    def correct_text(
        self,
        input_text: str,
        show_semantic_details: bool = True
    ) -> tuple:
        """
        Correct input text and return results
        
        Returns:
            (corrected_text, details_text, semantic_status)
        """
        if not input_text.strip():
            return "", "Please enter some Nepali text.", ""
        
        # Get correction with details
        result = self.pipeline.correct_with_details(input_text)
        
        corrected = result['correction']
        
        # Build details text
        details = []
        details.append("### Correction Details\n")
        
        if result['changes_made']:
            details.append("✅ **Changes made**")
        else:
            details.append("✅ **No errors detected**")
        
        # Semantic validation
        if show_semantic_details:
            val = result['semantic_validation']
            details.append("\n### Semantic Validation\n")
            
            if val['is_plausible']:
                details.append("✅ **Semantically correct**")
                semantic_status = "✅ Plausible"
            else:
                details.append(f"⚠️ **Semantic issues detected ({val['severity']})**")
                semantic_status = f"⚠️ {val['severity'].title()}"
                
                for issue in val['issues']:
                    details.append(f"  - {issue}")
        else:
            semantic_status = "Not checked"
        
        details_text = "\n".join(details)
        
        return corrected, details_text, semantic_status
    
    def batch_correct(self, input_texts: str) -> str:
        """Correct multiple sentences (one per line)"""
        if not input_texts.strip():
            return "Please enter sentences (one per line)"
        
        lines = [line.strip() for line in input_texts.split('\n') if line.strip()]
        
        if not lines:
            return "No valid sentences found"
        
        corrections = self.pipeline.correct_batch(lines, batch_size=8)
        
        # Format output
        output = []
        output.append("### Batch Correction Results\n")
        
        for i, (orig, corr) in enumerate(zip(lines, corrections), 1):
            output.append(f"**{i}. Original:** {orig}")
            output.append(f"   **Corrected:** {corr}")
            
            if orig != corr:
                output.append("   ✅ Changed")
            else:
                output.append("   ℹ️ No changes")
            
            output.append("")
        
        return "\n".join(output)
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .output-text {
            font-size: 18px;
            font-weight: bold;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Nepali Grammar Correction") as demo:
            gr.Markdown(
                """
                # 🇳🇵 Nepali Semantic-Aware Grammar Error Correction
                
                This system corrects grammatical errors in Nepali text and validates semantic correctness.
                
                **Features:**
                - Corrects 14 types of errors (orthographic, morphological, syntactic, semantic)
                - Detects semantic anomalies (e.g., "milk gives cow")
                - Handles multi-error sentences
                """
            )
            
            with gr.Tab("Single Sentence"):
                gr.Markdown("### Correct a single Nepali sentence")
                
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Input Text (Nepali)",
                            placeholder="म किताब पढछु",
                            lines=3
                        )
                        
                        show_semantic = gr.Checkbox(
                            label="Show semantic validation details",
                            value=True
                        )
                        
                        correct_btn = gr.Button("Correct Text", variant="primary")
                    
                    with gr.Column():
                        output_text = gr.Textbox(
                            label="Corrected Text",
                            lines=3,
                            elem_classes=["output-text"]
                        )
                        
                        semantic_status = gr.Textbox(
                            label="Semantic Status",
                            lines=1
                        )
                        
                        details_text = gr.Markdown(label="Details")
                
                # Examples
                gr.Examples(
                    examples=[
                        ["म किताब पढछु"],  # Missing halant
                        ["उनी घर मा छ"],  # Wrong agreement
                        ["किताबले खाना खान्छ"],  # Semantic error
                        ["तपाईं कहा जानुहुन्छ"],  # Missing halant
                        ["बच्चा ले दुध खानछ"],  # Spacing + halant
                    ],
                    inputs=input_text,
                )
                
                correct_btn.click(
                    fn=self.correct_text,
                    inputs=[input_text, show_semantic],
                    outputs=[output_text, details_text, semantic_status]
                )
            
            with gr.Tab("Batch Correction"):
                gr.Markdown("### Correct multiple sentences at once")
                
                with gr.Row():
                    with gr.Column():
                        batch_input = gr.Textbox(
                            label="Input Sentences (one per line)",
                            placeholder="म किताब पढछु\nउनी घर मा छ\nतपाईं कहा जानुहुन्छ",
                            lines=10
                        )
                        
                        batch_btn = gr.Button("Correct All", variant="primary")
                    
                    with gr.Column():
                        batch_output = gr.Markdown(label="Results")
                
                batch_btn.click(
                    fn=self.batch_correct,
                    inputs=batch_input,
                    outputs=batch_output
                )
            
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ## About This System
                    
                    **Model Details:**
                    - Base Model: mT5-small (300M parameters)
                    - Fine-tuning: QLoRA (4-bit, ~4.2M trainable parameters)
                    - Training Data: 10,000+ Nepali error-correction pairs
                    - Semantic Validator: NepBERTa-based plausibility classifier
                    
                    **Error Types Handled:**
                    
                    **Orthographic:**
                    - Raswa/Dirgha vowel confusion
                    - Conjunct consonant errors
                    - Spacing errors
                    
                    **Morphological:**
                    - Case marker errors
                    - Verb agreement errors
                    - Tense errors
                    
                    **Syntactic:**
                    - Word order violations
                    - Missing/extra words
                    
                    **Semantic:**
                    - Selectional preference violations
                    - Honorific mismatches
                    - Entity type incompatibilities
                    
                    **Citation:**
                    ```
                    @article{yourname2025nepali,
                      title={Semantic-Aware Grammar Error Correction for Nepali},
                      author={Your Name},
                      year={2025}
                    }
                    ```
                    
                    **GitHub:** [github.com/yourusername/nepali-semantic-gec](https://github.com/yourusername/nepali-semantic-gec)
                    """
                )
        
        return demo


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nepali GEC Web Demo")
    parser.add_argument(
        '--model_path',
        type=str,
        default='./outputs/nepali_gec_model',
        help='Path to trained model'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public share link'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run on'
    )
    
    args = parser.parse_args()
    
    # Initialize demo
    try:
        demo_app = NepaliGECDemo(model_path=args.model_path)
        interface = demo_app.create_interface()
        
        # Launch
        print(f"\n{'='*60}")
        print("Launching Nepali GEC Demo...")
        print(f"{'='*60}\n")
        
        interface.launch(
            share=args.share,
            server_port=args.port,
            server_name="0.0.0.0"
        )
        
    except Exception as e:
        print(f"Error launching demo: {e}")
        print("\nMake sure:")
        print("  1. Model is trained and saved")
        print("  2. Gradio is installed: pip install gradio")
        print(f"  3. Model path is correct: {args.model_path}")


if __name__ == "__main__":
    # Check if gradio is installed
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "gradio"])
        import gradio as gr
    
    main()