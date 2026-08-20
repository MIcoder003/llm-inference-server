# llm-inference-server

A continuous-batching LLM inference engine written in C++ on top of [llama.cpp](https://github.com/ggml-org/llama.cpp).

The goal of this project is to build the scheduling layer that serving engines like vLLM and TensorRT-LLM provide — request admission, per-sequence KV cache management, and slot reuse — from the ground up, rather than to reimplement the transformer itself. Model execution is delegated to llama.cpp; everything above `llama_decode` is written here.

---

## Why batching matters

A single forward pass loads the entire model's weights from memory. During **prefill**, that cost is spread across every token in the prompt. During **decode**, the same cost produces exactly one token. Decode is therefore memory-bandwidth-bound and leaves the hardware largely idle.

Measured on this project with a 1B model on CPU:

| Phase   | Throughput   |
| ------- | ------------ |
| Prefill | ~17-21 tok/s |
| Decode  | ~10-11 tok/s |

Since a decode step drags the weights through memory regardless, the fix is to carry as many sequences as possible on that same pass. Adding a sequence costs one extra sampling call — around 0.3% of the work of a forward pass — and rides the existing matmuls for free.

That asymmetry is what this project exploits.

---

## Build

Requires CMake 3.14+ and a C++17 compiler.

```bash
git clone <this-repo>
cd llm-inference-server
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp

cmake -B build
cmake --build build
```

Place a GGUF model at `models/` and update `model_path` in `src/main.cpp`. Development used `gemma-3-1b-it-Q4_K_M.gguf`.

## Run

```bash
./build/llm-inference-server [n_seq_max] [n_predict]
```

- `n_seq_max` — concurrent sequence slots (default 3)
- `n_predict` — max tokens generated per request (default 5)

Run from the project root; the model path is relative.

---

## Architecture

### Sequence state

Each request is tracked as a `Sequence`:

```cpp
struct Sequence {
    int                      seq_id;        // KV cache slot, assigned at admission
    std::string              prompt;
    std::vector<llama_token> tokens;        // prompt + generated
    int                      n_past;        // position within this sequence
    int                      batch_idx;     // where this seq's logits sit in the batch
    int                      n_decode;
    bool                     done;
    bool                     needs_prefill;
    std::string              output;
};
```

Two fields carry most of the complexity:

**`seq_id`** tags every token handed to llama.cpp. The library keeps a separate KV cache region per `seq_id` and builds the attention mask so sequences cannot see each other's tokens. Isolation is enforced entirely through this tag.

**`batch_idx`** records which flat slot in the batch holds this sequence's logits. It must be re-recorded every iteration, because the batch shrinks as sequences retire — if sequence 0 finishes, sequence 1 slides from slot 1 to slot 0. Caching a stale index yields plausible-looking but incorrect output, with no crash.

### The scheduler loop

```
while (waiting or running):

    ADMIT
        budget = n_batch - len(running)          # each running seq costs 1 decode token
        while waiting and free_slots:
            if next.prompt_len > budget: break   # all-or-nothing, no chunking
            assign seq_id from free_slots
            mark needs_prefill
            move waiting -> running

    BUILD BATCH
        for each running sequence:
            if needs_prefill:  add all prompt tokens, logits on the last one only
            else:              add one token at pos = n_past

    DECODE
        one llama_decode call serves every sequence

    SAMPLE
        for each running sequence: sample at its own batch_idx
        mark done on EOG or token limit

    RETIRE
        llama_memory_seq_rm(...)   # free the KV cache
        return seq_id to free_slots
        move running -> finished
```

### Two resources, two granularities

Admission is gated by two independent checks, and conflating them is the usual source of confusion:

- **Sequence slots** are request-level. A request holds a whole slot from admission to retirement. `n_seq_max` caps the count.
- **The token budget** is iteration-level. Each decoding sequence contributes one token; a prefilling sequence contributes its whole prompt. `n_batch` caps the total.

A request needs both to be admitted. Under the all-or-nothing policy implemented here, a prompt larger than the remaining budget waits an entire iteration rather than being split.

### KV cache release

When a sequence finishes, two separate things are freed:

1. **The slot** — bookkeeping in this program; the `seq_id` returns to `free_slots`.
2. **The KV cache** — a real call into llama.cpp:
   ```cpp
   llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
   ```

Skipping the second is the defining bug of this phase. The next request to occupy that slot starts at position 0 while stale entries still sit at positions 0..N. Attention reads both. The result is grammatical, on-topic, and quietly wrong.

---

## Development phases

**Phase 1 — single sequence.** Load model, tokenize, prefill, greedy decode loop, detokenize. Establishes a correctness baseline and exposes the prefill/decode throughput gap that motivates everything after.

**Phase 2 — static batching.** Multiple prompts in one batch, all starting together and running until the last finishes. Introduces `seq_id`, per-sequence position tracking, and per-sequence logits indexing. The batch only shrinks; a request arriving mid-flight has nowhere to go.

**Phase 3 — continuous batching.** Requests arrive into a `waiting` queue and are admitted to free slots as they open. Prefill moves inside the loop, so a newly admitted request prefills in the same `llama_decode` call where existing requests are decoding. Finished sequences release their KV cache and return their slot.

Sample output showing slot reuse:

```
[iter 1] admit slot 0 <- "The capital of France is"
[iter 1] admit slot 1 <- "The capital of India is"
[iter 1] admit slot 2 <- "The capital of Spain is"
[iter 6] retire slot 0
[iter 7] admit slot 0 <- "The capital of Japan is"
```

---

## Benchmark

Six requests, `n_predict = 5`, varying only the slot count:

| `n_seq_max` | Wall time | Throughput | Iterations |
| ----------- | --------- | ---------- | ---------- |
| 1           | _TBD_     | _TBD_      | _TBD_      |
| 3           | _TBD_     | _TBD_      | _TBD_      |

Model load time is excluded; it is a one-time disk read unrelated to scheduling.

---

## Not yet implemented

- **Chunked prefill.** A prompt exceeding the remaining token budget currently waits a full iteration. Slicing it would let admission proceed immediately and stop long prompts from stalling decode.
- **Preemption.** No eviction path when the KV cache is exhausted.
- **Paged KV cache.** Slots are fixed-size regions of the context. Block-level allocation would remove the worst-case reservation per sequence.
- **Prefix caching.** Shared prompt prefixes are prefilled once per request.
- **Per-sequence samplers.** A single greedy sampler is shared. Any stateful sampler (repetition penalty, seeded RNG) would leak state across sequences.
- **Network layer.** Requests come from a hardcoded list. An HTTP frontend would push into the same `waiting` queue.
