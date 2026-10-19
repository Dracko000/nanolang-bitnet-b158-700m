---
license: mit
language:
  - en
library_name: nanow
pipeline_tag: text-generation
base_model: 1bitLLM/bitnet_b1_58-large
base_model_relation: quantized
tags:
  - ternary
  - bitnet
  - 1.58-bit
  - quantization
  - cpu-inference
  - nanow
  - agent
---

# NanoLang BitNet b1.58 700M — `.nanow` container

**A 1.58-bit ternary 700M LLM packed into a single 350 MiB file that decodes at
~22–26 tok/s on four CPU cores, ~760 MB of RAM, no PyTorch, no GPU.**

This repository hosts the **`.nanow` container** of the BitNet b1.58 700M
checkpoint, pre-quantized to ternary and ready for the
[NanoLang](https://github.com/Dracko000/nanolang) runtime.

---

## Read this first: what this release is

This is **not a newly trained model**. It is the upstream
[`1bitLLM/bitnet_b1_58-large`](https://huggingface.co/1bitLLM/bitnet_b1_58-large)
checkpoint (MIT) **re-formatted** into the NanoLang `.nanow` container:

| | Upstream checkpoint | This release |
| :--- | :--- | :--- |
| Weights | fp32 master weights, quantized to ternary *at load time* | **ternary already applied and 2-bit packed** |
| File | `model.safetensors`, 2.72 GiB | `bitnet700m.nanow`, **350 MiB** |
| Tensors | 290 all-fp32 | 266 — **168 ternary**, 98 fp32 |
| Runtime needed | `torch` + `transformers` | `numpy` + `ctypes` only |

**Quality is therefore identical to the upstream model** — this container does not
improve or degrade the weights. The contribution is the container format and the
CPU runtime around them. The upstream project's license and attribution are
preserved; see [Attribution](#attribution).

Reproducible from source:

```bash
python3 convert_hf_to_nanow.py --hf 1bitLLM/bitnet_b1_58-large \
    --out models/bitnet700m/bitnet700m.nanow
```

---

## Model details

| | |
| :--- | :--- |
| **Base model** | [`1bitLLM/bitnet_b1_58-large`](https://huggingface.co/1bitLLM/bitnet_b1_58-large) |
| **Paper** | *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) |
| **Training data** | RedPajama, 100B tokens |
| **Parameters** | **728.8 M** total — 679.5 M ternary (2-bit) + 49.4 M fp32 (embedding + norms) |
| **Architecture** | Llama-style transformer, RMSNorm, RoPE Θ=10000, SwiGLU, tied embeddings |
| **Layers / hidden** | 24 / 1536 |
| **Heads** | 16 query · 16 key/value · head_dim 96 |
| **Intermediate** | 4096 |
| **Vocab** | 32,002 (SPM) |
| **Context** | 2,048 tokens |
| **Weight precision** | ternary `{-1, 0, +1}`, one f32 scale per row |
| **Container** | `nanow` v1 — mmap-able, self-describing |
| **License** | MIT |

### Upstream evaluation (reported by 1bitLLM, not re-run here)

| Model | PPL | ARC-e | ARC-c | HellaSwag | BoolQ | OpenBookQA | PIQA | WinoGrande | Avg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 700M (reported) | 12.33 | 54.7 | 23.0 | 37.0 | 60.0 | 20.2 | 68.9 | 54.8 | 45.5 |
| **BitNet b1.58 700M (reproduced)** | **12.78** | 51.4 | 21.8 | 35.0 | 59.6 | 20.6 | 67.5 | 55.4 | **44.5** |

These are the checkpoint authors' numbers and are reproduced verbatim from the
upstream card. This release does not re-run the harness and does not claim them as
its own measurements.

---

## Container contents

`MANIFEST.json` lists every tensor with its logical shape, storage dtype, packed
size and scale. Summary:

| Group | Tensors | Params | Bits/weight | Bytes |
| :--- | ---: | ---: | ---: | ---: |
| ternary (all linear projections) | 168 | 679.5 M | 2 | **169.9 MB** |
| float32 (embedding + all norms) | 98 | 49.4 M | 32 | 197.5 MB |
| **total** | **266** | **728.8 M** | — | **367.5 MB** |

```
bitnet700m.nanow   367,513,600 bytes   sha256 97e9281739c2488f…
```

> **Honest note on size.** Half the file is the **fp32 token embedding**
> (32,002 × 1,536 = 49.2 M params, 197 MB) because the weights are tied and stored
> in fp32. The ternary weights themselves are only 170 MB. Quantizing the
> embedding to int8 would roughly halve the container, and is a listed TODO, not
> a shipped feature.

Verify the download:

```bash
python3 verify.py           # checks CHECKSUMS.txt against the files present
```

---

## Measured performance

Reference hardware — a modest 4-core CPU, chosen so the numbers mean something on
a laptop or a cheap VPS:

| | |
| :--- | :--- |
| CPU | AMD EPYC, **4 cores @ 2.0 GHz** |
| ISA | `avx avx2 fma bmi1 bmi2 f16c` — **no AVX-512, no VNNI, no AMX** |
| Cache | L1d 128 KB · L2 2 MB · L3 8 MB |
| RAM | 8 GB, no swap |
| GPU | none |

| Metric | Value |
| :--- | ---: |
| **Decode** | **22–26 tok/s** |
| Prefill (18-token prompt) | 33 tok/s |
| **Peak RSS** | **754–760 MB** |
| Effective DRAM read | 7.7–9.1 GB/s |
| First working version, same box | 1.14 tok/s → **~20× faster** |

The 350 MiB container defines a hard decode roofline — one full sweep of the
weights per token. At 22–26 tok/s that is 7.7–9.1 GB/s of effective bandwidth, so
the runtime is operating close to what this memory system can deliver. There is
little headroom left without shrinking the weights further (see the embedding
note above) or moving to a wider memory bus.

Run-to-run variance on this shared machine is ±5–10 %; figures near the noise
floor are reported as such rather than rounded up.

---

## Features

Everything below is implemented in the engine under
[`Dracko000/nanolang`](https://github.com/Dracko000/nanolang).

### Ternary-native execution

The inner loop is not a floating-point multiply-add. Ternary weights let the
linear layer become **conditional add / subtract / skip** over dynamically
quantized INT8 activations, accumulated in INT32, with one rescale per output row.
On CPUs without VNNI or AMX — most laptops, most cheap VPS — this is what makes a
700M model usable without a GPU.

### Constrained tool calling

A 700M model free-generates well-formed tool calls poorly. Measured on a 10-tool
selection task:

| Mode | Parsable | Correct tool |
| :--- | ---: | ---: |
| Free generation, full prompt | 5/8 | 5/8 |
| Free generation, compact prompt | 1/8 | 1/8 |
| **Constrained decoding** | — | **9/10** |

So structure is never left to chance: the **tool name is masked at the logit
level**, arguments are forced to their schema, and only the *values* are free. The
model chooses; the runtime guarantees the syntax.

### Online grounding

`--online` retrieves from Wikipedia and DuckDuckGo using the **standard library
only** (no `requests`, no HTTP framework) and injects the snippets into the
context. The model is online, not a frozen snapshot.

### Prefix KV reuse — across runs

The agent preamble is a static prefix, and its KV state is a pure function of
token ids and weights, so it is safe to serialize to `.npz`:

| | prefill | wall clock |
| :--- | ---: | ---: |
| No cache | 444 tokens every run | 45.7 s |
| Cold run (writes 131 MB) | 444 tokens | 56.6 s |
| **Warm run (loads cache)** | 0 tokens | **13.5 s** |
| Warm run, compact prefix (172 tokens, 51 MB) | 0 tokens | **8.0 s** |

A 3-step file-manipulation agent task runs in **87.1 s with the cache versus
189.4 s without** (2.2×).

### Agent loop with a sandbox

Ten tools (`list_dir`, `read_file`, `write_file`, `append_file`, `delete_file`,
`make_dir`, `move_file`, `http_get`, `web_search`, `run_bash`) behind typed
schemas, a path sandbox that resolves symlinks, a bash deny-list with a timeout
and an output cap, duplicate-call detection, and a multi-step planner that splits
only on explicit sequencing connectors (*then*, *after that*) — never a bare *and*.

---

## Why this container, versus the alternatives

Honest comparison of **deployment characteristics**. The only column measured by
us is the first; the rest are structural facts about the alternatives.

| | `bitnet700m.nanow` (this) | Same weights, `transformers` | `llama.cpp` + ternary GGUF |
| :--- | :--- | :--- | :--- |
| Weights on disk | **350 MiB** | 2.72 GiB fp32 | ~200–350 MiB |
| Ternary applied | **at conversion** | at load, every run | at conversion |
| Runtime deps | **numpy + ctypes** | torch + transformers | C++ binary |
| Peak RAM | **~760 MB** | not measured here | ~450–600 MB (typical) |
| Decode on 4 CPU cores | **22–26 tok/s** | not measured here | typically faster |
| Readable end-to-end | **~2.4 k lines, all of it** | framework internals | large C++ codebase |
| Tool / agent layer | **built in** | DIY | DIY |
| Constrained decoding | **built in** | DIY | partial (GBNF) |

**Where llama.cpp wins, plainly:** it is more mature, better optimised, and faster
for raw generation. If you only need tokens per second, use it.

**Where this is different:** the whole stack — tokenizer, attention, ternary
kernel, sampler, constrained decoder, tool loop, grounding — is a few thousand
lines you can read in an afternoon, and the agent/tooling layer is part of the
runtime rather than something you bolt on. It is a learning-grade and
hackable-grade engine that happens to be fast enough to be useful on four cores.

**Against dense fp16 small models** (`SmolLM2-135M`, `Qwen2.5-0.5B`): those are
different models with different training and different quality. Nothing here
claims to answer *better* than they do. The comparison that matters is
deployment: a diffusion-free, framework-free, 350 MiB CPU artifact versus an
fp16 model that needs a tensor framework and several times the RAM for a similar
parameter count.

---

## How to run it

```bash
git clone https://github.com/Dracko000/nanolang && cd nanolang
make                                    # builds phase0/bin/libternary.so

mkdir -p models/bitnet700m && cd models/bitnet700m
huggingface-cli download dracko14/nanolang-bitnet-b158-700m --local-dir .
cd ../..

python3 run_smollm2.py \
  --model models/bitnet700m/bitnet700m.nanow \
  --config models/bitnet700m/config.json \
  --tokenizer models/bitnet700m/tokenizer.json \
  --prompt "Explain what a ternary weight is, briefly." \
  --tokens 120 --temp 0.7
```

Agent mode, online and grounded:

```bash
python3 run_smollm2.py \
  --model models/bitnet700m/bitnet700m.nanow \
  --config models/bitnet700m/config.json \
  --tokenizer models/bitnet700m/tokenizer.json \
  --agent --constrained-args --planner route --online \
  --workspace ./nanow_workspace --kv-cache-dir ./.kv \
  --prompt "create todo.txt with the content buy milk then read it" --max-steps 3
```

The runner turns that into `write_file(todo.txt, "buy milk")` then
`read_file(todo.txt)`. Note that only the tool calls are reliable — the model's
free-generated closing prose is not, because this checkpoint is not
instruction-tuned.

---

## Intended use and out of scope

**Intended:** on-device and edge CPU inference; teaching how ternary LLMs work end
to end; local agents that run filesystem and web tasks on a machine without a GPU;
as a compact base for constrained-generation experiments.

**Out of scope:** anything needing high factual accuracy or long-form reasoning —
this is a 700M model and it behaves like one. Do not use it for medical, legal,
financial, or safety-critical decisions. It is not instruction-tuned; it responds
to prompts, not to a chat template.

---

## Limitations

- **700M-parameter quality.** Coherent for short and medium generations; it will
  invent facts, lose the thread on long outputs, and mangle multi-step reasoning.
- **Reliable tool calls, unreliable narration.** Under constrained decoding the
  tool call is correct, but the closing prose the model writes afterwards often
  drifts. Trust the tool transcript, not the summary.
- **Not instruction-tuned.** No RLHF/DPO/SFT stage. Treat it as a base model.
- **Free-generation tool calls are unreliable** (5/8 best case). Use
  `--constrained-args`.
- **Elementwise C kernels buy ~5 % end-to-end.** They are 1.9×–2.6× faster per
  op, but the ternary GEMV is ~95 % of a token.
- **Half the container is the fp32 embedding.** Known and listed, not yet fixed.
- **No GPU/NPU backend is wired to this runtime.** Mobile sources exist in
  `phase2/`; they are not exercised by the quickstart.
- **Bash is deny-listed, not sandboxed.** Do not enable `--allow-bash` on a
  machine you care about.
- **`--online` injects retrieved text without citing it.** The model can still
  misattribute.

---

## Attribution

- **Weights:** [`1bitLLM/bitnet_b1_58-large`](https://huggingface.co/1bitLLM/bitnet_b1_58-large),
  MIT, a reproduction of *The Era of 1-bit LLMs* ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)),
  trained on RedPajama for 100B tokens.
- **Container and runtime:** [NanoLang](https://github.com/Dracko000/nanolang), MIT.
- This is an **independent, community** release. It is **not affiliated with,
  endorsed by, sponsored by, or authorized by** the authors of BitNet, Microsoft,
  the 1bitLLM project, or Hugging Face.

If you use these weights, cite the original work:

```bibtex
@misc{ma2024era1bitllms,
  title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits},
  author={Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang
          and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei},
  year={2024},
  eprint={2402.17764},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
