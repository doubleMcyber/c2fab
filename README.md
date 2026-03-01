# C²FAB: Vector Causal Charge-Field Attention Bias

A novel, physics-inspired attention mechanism built for **Ministral-8B**, developed over the course of a 2-day hackathon.

## Overview
Long-context models frequently suffer from "signal dilution"—losing track of distant but vital evidence amid thousands of distractor tokens. Current solutions either brutally compress the KV cache or rely on external RAG infrastructure. 

**V-C²FAB** introduces a plug-and-play attention control module that lives natively inside the Transformer. We treat important tokens as "charges" that emit a causal, multi-dimensional potential field across the sequence. At query time, a "receptor" reads this field, dynamically reshaping the attention energy landscape to pull focus toward the buried evidence.

### Core Features
- **Plug-and-Play:** Operates purely by monkey-patching the top 4 layers of the LLM. No retraining of the base model weights is required.
- **Continuous-Space Retrieval:** Employs a learnable Infinite Impulse Response (IIR) filter to propagate attention bias O(T) without breaking inference mechanics.
- **Fast Training:** Utilizes a contrastive (InfoNCE) weak-supervision objective on layer 22 hidden states, allowing the bias heads to be trained in hours on a single GPU (or Apple Silicon `mps`).

## Project Status
Currently in active development.

- [ ] **Phase 0:** Mathematical parallelization of causal IIR fields.
- [ ] **Phase 1:** Synthetic evidence-localization dataset generation.
- [ ] **Phase 2:** Charge & Receptor MLP implementation.
- [ ] **Phase 3:** Contrastive Distillation training loop.
- [ ] **Phase 4:** Attention-layer monkey patching (Top 4 layers).
- [ ] **Phase 5:** LongBench QA evaluation.

## License
MIT License / Mistral Research License (depending on model usage).
