# PROJECT CONTEXT: V-C²FAB (Vector Causal Charge-Field Attention Bias)

## 1. Project Identity & Goal
We are building a plug-and-play attention control module for **Ministral-8B-Instruct-2410**. 
**The Problem:** Long-context LLMs suffer from signal dilution, ignoring distant but vital evidence.
**The Solution:** V-C²FAB. We train a lightweight "charge head" to identify evidence and emit a multi-dimensional causal vector field. A "receptor head" reads this field at query time to dynamically bias attention logits, effectively doing continuous-space retrieval inside the attention mechanism.

## 2. Core Architecture & Math
Ministral-8B has 36 layers. We are modifying Scaled Dot-Product Attention: `Attn = softmax((Q @ K^T)/sqrt(d) + Bias) @ V`
We inject `Bias` dynamically into the **top 4 layers only** (e.g., layers 32, 33, 34, 35).

Let $x_u$ be the hidden state of key/value token $u$ **extracted from a configurable mid-layer (default Layer 22 out of 36)** to capture semantic layout without over-fitting to next-token syntax.
Let $x_q$ be the hidden state of query token $q$.
Let $D$ be the dimension of our vector field (e.g., $D=8$).

*   **1. Charge Head:** $C_u = \text{ReLU}(W_2 \cdot \text{GELU}(W_1 \cdot \text{LayerNorm}(x_u))) \in \mathbb{R}^D$ 
    * *Regularization:* Apply L1 penalty to $C_u$ to enforce sparsity.
*   **2. Field Propagation (IIR):** $\Phi_u^{(d)} = \lambda_d \Phi_{u-1}^{(d)} + C_u^{(d)}$
    * where $\lambda \in (0,1)$ is learnable per dimension.
*   **3. Receptor Head:** $R_q = \text{MLP}(x_q) \in \mathbb{R}^D$
*   **4. Bias Score:** $s_{q,u} = R_q \cdot \Phi_u$
    * Final injected bias: $Bias_{q,u} = \alpha \cdot s_{q,u}$ (where $\alpha$ is a learned per-layer scalar).

## 3. The Training Pipeline & Primary Benchmarks
We use Weakly Supervised Evidence Matching.
*   **Base Model:** Ministral-8B is entirely frozen.
*   **Loss Function (InfoNCE):**
    $\mathcal{L} = -\log \frac{\sum_{u \in E}\exp(s_{q,u}/\tau)}{\sum_{u \in E}\exp(s_{q,u}/\tau) + \sum_{u \in N}\exp(s_{q,u}/\tau)}$
    * $E$ = Evidence tokens, $N$ = Distractors (~256/query), $\tau$ = ~0.1
*   **Primary Benchmarks:** 
    1. Synthetic distractor retrieval (matches training signal).
    2. LongBench QA subset.
    *(Note: Standard Needle-In-A-Haystack is secondary, as it tests numeric exact-match rather than semantic evidence).*

## 4. Strict Engineering Constraints (READ CAREFULLY)
*   **Environment:** User is on a 32GB Unified Memory MacBook. Code must support `mps` (Metal Performance Shaders) and `cuda`.
*   **Ministral-8B GQA Specs:** The model has 32 Query heads and 8 KV heads. Our computed Bias matrix `[Batch, Q_Len, KV_Len]` must be broadcasted to `[Batch, Num_Q_Heads, Q_Len, KV_Len]` safely.
*   **Attention Fallback Warning:** For Phase 4 monkey-patching, if we use Eager SDPA with chunked queries to inject bias, we will silently lose FlashAttention speeds. This is acceptable for hackathon evaluation, but keep chunk sizes strictly bounded to prevent OOMs.

## 5. Execution Phases (Onion Architecture)
Do not jump ahead. We will complete these sequentially:
*   **Phase 0 (Setup):** Implement `c2fab_math.py`. *Constraint exception: You may implement the IIR filter using a sequential `for` loop first for mathematical correctness. Then, write a parallel version (log-cumsum) and prove they match via unit tests.*
*   **Phase 1 (Data Gen):** Implement `data_gen.py`. Generate 8k-context synthetic examples with known evidence spans and binary masks.
*   **Phase 2 (The Model Core):** Implement `modules.py`. Create `C2FAB_Heads` (charge MLP, receptor MLP, lambdas). Implement the InfoNCE loss function.
*   **Phase 3 (Training Loop):** Implement `train.py`. Run forward passes of frozen Ministral up to Layer 22, extract $x_u$ and $x_q$, and train `C2FAB_Heads`. Monitor Recall@K and Sparsity %.
*   **Phase 4 (Monkey Patching):** Implement `patcher.py`. Connect the trained heads into Ministral's top 4 layers (Layers 32-35).
*   **Phase 5 (Evals & API):** Run LongBench QA / Synthetic Retrieval and wrap in `ChargeFieldMinistral` API.

## 6. Working Style & Metrics
*   Write modular, defensive PyTorch code.
*   Always type-hint and comment tensor shapes (e.g., `# x:[batch_size, seq_len, hidden_dim]`).
*   Wait for the user to specify which Phase/Step we are currently working on before writing code.

## 7. Overnight Training Constraints & Novelty (Disentangled Fields)
*   **Memory Bottleneck Fix (Forward Hooks):** DO NOT use `output_hidden_states=True`. It causes SSD swapping on MPS. Instead, register a PyTorch `forward_hook` on `model.model.layers[22]` to capture the hidden state, detach it immediately, and remove the hook. 
*   **MPS Optimization:** Use `device="mps"` and `dtype=torch.float16`. Wrap the entire LLM forward pass in `with torch.no_grad():`.
*   **The Novelty (Orthogonal Field Penalty):** To force the $D$ dimensions of our causal field to specialize (e.g., Dim 1 finds names, Dim 2 finds dates), we apply an orthogonality penalty across the sequence.
    * Let $C \in \mathbb{R}^{N \times D}$ be the flattened charge vectors across the sequence.
    * Normalize along the batch/sequence dimension so each feature column has unit length: `C_norm = F.normalize(C, p=2, dim=0)`
    * Compute the Gram matrix: `Gram = C_norm.T @ C_norm` (shape $D \times D$)
    * Penalty: `ortho_loss = torch.norm(Gram - torch.eye(D), p='fro')`
    * Final Loss: `loss = infonce_loss + (0.01 * L1_sparsity) + (0.05 * ortho_loss)`