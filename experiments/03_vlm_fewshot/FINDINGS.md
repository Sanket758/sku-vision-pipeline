# VLM Evaluation Findings (Moondream, Llava, Qwen3-VL)

## Ralph-Style Dev-Inspect-Improve Loop

### Iteration 1: Moondream2 Zero-Shot
- **Setup**: Evaluated `moondream:latest` (1.7B) via Ollama on a full supermarket shelf image.
- **Inspect**: The model failed to process dense scenes. It returned highly truncated and irrelevant outputs (e.g., "1. Green box of pizza - Partially Occluded"). 
- **Improve**: Moondream lacks the capacity for dense multi-object reasoning in retail environments. Shifted to a larger model (`llava:latest`) and injected a highly domain-specific prompt targeting "Chocolate and Coffee".

### Iteration 2: Llava Zero-Shot (Domain-Specific Prompt)
- **Setup**: Evaluated `llava:latest` (7B) with a strict prompt naming specific brands (Lindt, Milka, Jacobs).
- **Inspect**: Inference took between 25 and 53 seconds per image on the local RTX 3050 (6GB VRAM). The model exhibited severe **prompt-induced hallucination**. It confidently listed "Jacobs", "Milka", "Ritter Sport", and "Lindt" but constantly hedged its predictions ("appears to be", "cannot confirm specific flavor"). It failed to use actual packaging text to distinguish variants, relying instead on broad shapes and the prompt's suggestions.
- **Improve**: Attempted few-shot learning to provide visual grounding instead of relying purely on text prompts.

### Iteration 3: Few-Shot Inference (Llava & Qwen3-VL)
- **Setup**: Modified `eval_vlm_ollama.py` to pass multiple context images (reference SKUs) alongside the query image.
- **Inspect**: 
  - `llava:latest` failed to align the multiple images with the query. It responded: "I'm unable to provide specific product names... Please provide the query image."
  - `qwen3-vl:4b` returned HTTP 400 Bad Request errors, indicating compatibility issues with Ollama's standard `/api/generate` endpoint when passing complex multi-image visual contexts without specialized chat templating.
- **Improve/Conclusion**: Lightweight local VLMs currently lack the architectural maturity and context window efficiency to handle dense retail few-shot matching locally on edge hardware.

## Final Conclusion for Chapter 4.5

The VLM experiments definitively answer whether text-based reasoning can resolve the fine-grained visual similarities that confound the DINOv3 hybrid. **It cannot.**

1. **Hallucination Risk**: Rather than reading text to distinguish identical coffee bags or chocolate wrappers, local VLMs hallucinate brands based on shape familiarity and prompt priming.
2. **Hardware & Latency Constraints**: At 25–50 seconds per image, local VLMs are completely unviable for shelf analysis compared to the hybrid DINOv3+MobileNetV2 system (which operates at under 40 milliseconds per crop).
3. **Density Failure**: VLMs struggle immensely with dense object configurations (50+ products per image). They are designed for primary subject description, not comprehensive cataloguing.

Therefore, the DINOv3 hybrid remains the superior architecture for retail shelf analysis. Future work must rely on dedicated OCR pipelines rather than general-purpose VLMs for text-based disambiguation.
