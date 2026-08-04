```{eval-rst}
.. index::
    single: stateful model
    single: multifunction model
    single: transformer
    single: prefill decode
```

# Convert Auto-Regressive Transformers with Prefill / Decode Multifunctions

Auto-regressive transformer language models typically run in two distinct
phases:

- **Prefill** — process the prompt of length `N` in a single forward pass to
  populate the KV cache.
- **Decode** — generate the next token by passing a single token ID
  (`q_len = 1`) through the model and writing it to the next KV cache slot.

These phases share the same weights but have different input shapes, so a
single Core ML model with a fixed query length is not a great fit. This guide
shows how to combine [stateful KV cache](stateful-models) with
[multifunction models](multifunction-models) to get a single
`.mlpackage` that exposes both `prefill` and `decode` functions, deduplicates
the weights, and shares a single KV state across calls.

## Why split prefill and decode

A typical decoder-only LLM serves a 4 K-token prompt in a few hundred
milliseconds at prefill (large `q_len`, GPU/ANE matmul dominated) and
generates each subsequent token in a few milliseconds at decode (`q_len = 1`,
KV-cache read dominated). Tracing one model with a flexible `q_len` works
but forces every operation to handle the dynamic shape, which usually
prevents Core ML from compiling the kernels efficiently. Tracing two models
with different static `q_len` and merging them as a multifunction asset
avoids this problem and shares all the weights.

## End-to-end pattern

```{tip}
The toy model below is intentionally tiny so you can run the snippet on any
machine in seconds. The same pattern scales to production-size LLMs — see
the [HuggingFace `swift-transformers` Mistral 7B
example](https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/export.py)
referenced in [Stateful Models](stateful-models) for a real-world reference.
```

### Step 1 — Define a stateful KV-cache model

Register the K and V caches as buffers and write back to them with full-slice
assignment (`self.k_cache[:] = ...`). The Core ML torch frontend recognizes
this pattern and emits the corresponding `coreml_update_state` op.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_SIZE = 32
HEAD_DIM = 8
MAX_SEQ_LEN = 16
VOCAB_SIZE = 50


class ToyStatefulLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        self.q_proj = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.k_proj = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.v_proj = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.o_proj = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.lm_head = nn.Linear(EMBED_SIZE, VOCAB_SIZE, bias=False)
        self.register_buffer("k_cache", torch.zeros(1, MAX_SEQ_LEN, EMBED_SIZE))
        self.register_buffer("v_cache", torch.zeros(1, MAX_SEQ_LEN, EMBED_SIZE))

    def forward(self, input_ids, causal_mask, update_mask, k_pad_indices):
        x = self.embedding(input_ids)
        Q, K_new, V_new = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Scatter q_len rows of K_new/V_new into MAX_SEQ_LEN rows.
        K_padded = torch.index_select(K_new, 1, k_pad_indices.long())
        V_padded = torch.index_select(V_new, 1, k_pad_indices.long())

        # Mask-based merge with the existing cache.
        K = self.k_cache * (1 - update_mask) + K_padded * update_mask
        V = self.v_cache * (1 - update_mask) + V_padded * update_mask

        # Full-slice assignment triggers the stateful write.
        self.k_cache[:] = K
        self.v_cache[:] = V

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (HEAD_DIM ** 0.5)
        attn = F.softmax(scores + causal_mask, dim=-1)
        return self.lm_head(self.o_proj(torch.matmul(attn, V)))
```

### Step 2 — The mask-based dynamic-position write

The toy in [Stateful Models](stateful-models) updates the cache with a
plain Python slice assignment:

```python
k_cache[:, past_kv_len:end_step, :] = newly_computed_k
```

`torch.jit.trace` records `past_kv_len` and `end_step` as Python ints, so the
trace is only valid for one specific position. To support a dynamic write
position (necessary for real auto-regressive decode), gate the write with a
mask instead:

- `update_mask` has shape `(1, MAX_SEQ_LEN, 1)` with `1.0` at every row
  that should be overwritten and `0.0` everywhere else.
- `k_pad_indices` (shape `(MAX_SEQ_LEN,)`) gathers the freshly computed
  rows of `K_new` (`q_len` of them) into the `MAX_SEQ_LEN` slots — rows
  whose `update_mask` is 0 are gathered from index 0 and discarded by the
  mask.

This composition is what makes the same model definition usable for both
prefill (write rows 0..N-1) and decode (write row `current_pos`), with all
shapes static.

### Step 3 — Trace prefill and decode separately

```python
import numpy as np

PREFILL_QLEN = 4

def causal_mask(q_len, max_len, q_offset=0):
    m = torch.zeros(1, q_len, max_len)
    for i in range(q_len):
        for j in range(max_len):
            if j > i + q_offset:
                m[0, i, j] = float("-inf")
    return m

def update_mask(write_positions, max_len):
    m = torch.zeros(1, max_len, 1)
    for p in write_positions:
        m[0, p, 0] = 1.0
    return m

def k_pad_indices(write_positions, max_len):
    indices = torch.zeros(max_len, dtype=torch.int32)
    for new_row, dst_row in enumerate(write_positions):
        indices[dst_row] = new_row
    return indices

# --- Prefill: q_len=4, write rows [0..3] ---
m = ToyStatefulLLM().eval()
ids_p = torch.randint(0, VOCAB_SIZE, (1, PREFILL_QLEN), dtype=torch.int32)
ts_p = torch.jit.trace(m, (
    ids_p,
    causal_mask(PREFILL_QLEN, MAX_SEQ_LEN),
    update_mask(list(range(PREFILL_QLEN)), MAX_SEQ_LEN),
    k_pad_indices(list(range(PREFILL_QLEN)), MAX_SEQ_LEN),
))

# --- Decode: q_len=1, write row [4] ---
m_d = ToyStatefulLLM().eval()
m_d.load_state_dict(m.state_dict())
ids_d = torch.randint(0, VOCAB_SIZE, (1, 1), dtype=torch.int32)
ts_d = torch.jit.trace(m_d, (
    ids_d,
    causal_mask(1, MAX_SEQ_LEN, q_offset=4),
    update_mask([4], MAX_SEQ_LEN),
    k_pad_indices([4], MAX_SEQ_LEN),
))
```

Both traces target the same model class; only the static `q_len` differs.

### Step 4 — Convert each function to its own mlpackage

```python
import coremltools as ct

state_specs = [
    ct.StateType(
        wrapped_type=ct.TensorType(
            shape=(1, MAX_SEQ_LEN, EMBED_SIZE), dtype=np.float16
        ),
        name="k_cache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(
            shape=(1, MAX_SEQ_LEN, EMBED_SIZE), dtype=np.float16
        ),
        name="v_cache",
    ),
]

def shared_inputs(q_len):
    return [
        ct.TensorType(name="input_ids", shape=(1, q_len), dtype=np.int32),
        ct.TensorType(
            name="causal_mask", shape=(1, q_len, MAX_SEQ_LEN), dtype=np.float16
        ),
        ct.TensorType(
            name="update_mask", shape=(1, MAX_SEQ_LEN, 1), dtype=np.float16
        ),
        ct.TensorType(
            name="k_pad_indices", shape=(MAX_SEQ_LEN,), dtype=np.int32
        ),
    ]

mlmodel_p = ct.convert(
    ts_p,
    inputs=shared_inputs(PREFILL_QLEN),
    outputs=[ct.TensorType(name="logits", dtype=np.float16)],
    states=state_specs,
    minimum_deployment_target=ct.target.iOS18,
)
mlmodel_p.save("prefill.mlpackage")

mlmodel_d = ct.convert(
    ts_d,
    inputs=shared_inputs(1),
    outputs=[ct.TensorType(name="logits", dtype=np.float16)],
    states=state_specs,
    minimum_deployment_target=ct.target.iOS18,
)
mlmodel_d.save("decode.mlpackage")
```

### Step 5 — Merge into a single multifunction asset

```python
desc = ct.utils.MultiFunctionDescriptor()
desc.add_function(
    "prefill.mlpackage",
    src_function_name="main",
    target_function_name="prefill",
)
desc.add_function(
    "decode.mlpackage",
    src_function_name="main",
    target_function_name="decode",
)
desc.default_function_name = "prefill"
ct.utils.save_multifunction(desc, "llm.mlpackage")
```

`save_multifunction` deduplicates the embedding, attention, and LM-head
weights between the two functions because their hashes match — only the
function-specific shape information is duplicated.

### Step 6 — Run inference with shared state

Both functions are loaded from the same `.mlpackage` and a single state is
created that is passed across `predict` calls:

```python
prefill = ct.models.MLModel("llm.mlpackage", function_name="prefill")
decode = ct.models.MLModel("llm.mlpackage", function_name="decode")
state = prefill.make_state()

prefill.predict(
    {
        "input_ids": ids_p.numpy().astype(np.int32),
        "causal_mask": causal_mask(PREFILL_QLEN, MAX_SEQ_LEN).numpy().astype(np.float16),
        "update_mask": update_mask(list(range(PREFILL_QLEN)), MAX_SEQ_LEN).numpy().astype(np.float16),
        "k_pad_indices": k_pad_indices(list(range(PREFILL_QLEN)), MAX_SEQ_LEN).numpy().astype(np.int32),
    },
    state=state,
)

decode.predict(
    {
        "input_ids": ids_d.numpy().astype(np.int32),
        "causal_mask": causal_mask(1, MAX_SEQ_LEN, q_offset=4).numpy().astype(np.float16),
        "update_mask": update_mask([4], MAX_SEQ_LEN).numpy().astype(np.float16),
        "k_pad_indices": k_pad_indices([4], MAX_SEQ_LEN).numpy().astype(np.int32),
    },
    state=state,
)
```

The state is updated in place by each call, so the next decode step sees the
KV slots written by the previous one.

## Where to go from here

- The HuggingFace [Mistral 7B export
  example](https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/export.py)
  applies the same pattern at production scale.
- For details on the underlying primitives, see
  [Stateful Models](stateful-models) and
  [Multifunction Models](multifunction-models).
