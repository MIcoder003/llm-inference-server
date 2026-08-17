#include "llama.h"
#include "common.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), text.size(),
                       tokens.data(), tokens.size(), true, true) < 0) {
        fprintf(stderr, "tokenize: failed\n");
        exit(1);
    }
    return tokens;
}

std::string detokenize(const llama_vocab * vocab, llama_token id) {
    char buf[128];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) { fprintf(stderr, "detokenize: failed\n"); exit(1); }
    return std::string(buf, n);
}

struct Sequence {
    int                      seq_id        = -1;
    std::string              prompt;
    std::vector<llama_token> tokens;
    int                      n_past        = 0;
    int                      batch_idx     = -1;
    int                      n_decode      = 0;
    bool                     done          = false;
    bool                     needs_prefill = false;
    std::string              output;
};

int main(int argc, char* argv[]) {
    std::string model_path = "models/gemma-3-1b-it-Q4_K_M.gguf";

    // 6 requests, 3 slots -> three of these must queue
    std::vector<std::string> prompts{
        "The capital of France is",
        "The capital of India is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of Brazil is",
        "The capital of Kenya is"
    };

    int ngl        = 99;
    int n_predict  = 5;
    int n_seq_max  = 3;      // slot count — now a policy choice, not prompts.size()

    ggml_backend_load_all();

    // ---- model ----
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ---- queues ----
    std::vector<Sequence> waiting, running, finished;
    std::vector<int>      free_slots;

    for (int i = 0; i < (int) prompts.size(); i++) {
        Sequence s;
        s.prompt = prompts[i];
        s.tokens = tokenize(vocab, prompts[i]);
        waiting.push_back(s);          // no seq_id yet
    }

    for (int i = n_seq_max - 1; i >= 0; i--) free_slots.push_back(i);

    int n_longest = 0;
    for (auto & s : waiting) n_longest = std::max(n_longest, (int) s.tokens.size());

    int n_batch_size = n_longest + n_seq_max;   // one prefill + concurrent decodes

    // ---- context ----
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = n_seq_max * (n_longest + n_predict);
    ctx_params.n_batch   = n_batch_size;
    ctx_params.n_seq_max = n_seq_max;
    ctx_params.no_perf   = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    // ---- sampler ----
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_init(n_batch_size, 0, n_seq_max);

    // ---- scheduler loop ----
    const auto t_main_start = ggml_time_us();
    int iter = 0;

    while (!waiting.empty() || !running.empty()) {
        iter++;

        // --- ADMIT ---
        int budget = n_batch_size;
        budget -= (int) running.size();          // each running seq costs 1 decode token

        while (!waiting.empty() && !free_slots.empty()) {
            Sequence & next = waiting.front();
            if ((int) next.tokens.size() > budget) break;   // all-or-nothing, no chunking

            next.seq_id = free_slots.back();
            free_slots.pop_back();
            next.needs_prefill = true;
            budget -= (int) next.tokens.size();

            printf("[iter %d] admit slot %d <- \"%s\"\n",
                   iter, next.seq_id, next.prompt.c_str());

            running.push_back(next);
            waiting.erase(waiting.begin());
        }

        // --- BUILD BATCH ---
        common_batch_clear(batch);
        for (auto & s : running) {
            if (s.needs_prefill) {
                for (int i = 0; i < (int) s.tokens.size(); i++) {
                    bool is_last = (i == (int) s.tokens.size() - 1);
                    common_batch_add(batch, s.tokens[i], i, { s.seq_id }, is_last);
                    if (is_last) s.batch_idx = batch.n_tokens - 1;
                }
                s.n_past = s.tokens.size();
                s.needs_prefill = false;
            } else {
                common_batch_add(batch, s.tokens.back(), s.n_past, { s.seq_id }, true);
                s.batch_idx = batch.n_tokens - 1;
                s.n_past++;
            }
        }

        if (batch.n_tokens == 0) break;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }

        // --- SAMPLE ---
        for (auto & s : running) {
            llama_token tok = llama_sampler_sample(smpl, ctx, s.batch_idx);

            if (llama_vocab_is_eog(vocab, tok) || s.n_decode >= n_predict) {
                s.done = true;
                continue;
            }

            s.output += detokenize(vocab, tok);
            s.tokens.push_back(tok);
            s.n_decode++;
        }

        // --- RETIRE --- (backwards: erasing shifts indices)
        for (int i = (int) running.size() - 1; i >= 0; i--) {
            if (!running[i].done) continue;

            printf("[iter %d] retire slot %d\n", iter, running[i].seq_id);

            llama_memory_seq_rm(llama_get_memory(ctx), running[i].seq_id, -1, -1);
            free_slots.push_back(running[i].seq_id);

            finished.push_back(running[i]);
            running.erase(running.begin() + i);
        }
    }

    const auto t_main_end = ggml_time_us();

    // ---- results ----
    printf("\n");
    for (auto & s : finished) {
        printf("[seq %d] %s -->%s\n", s.seq_id, s.prompt.c_str(), s.output.c_str());
    }

    int n_decode_total = 0;
    for (auto & s : finished) n_decode_total += s.n_decode;

    const float t = (t_main_end - t_main_start) / 1000000.0f;
    fprintf(stderr, "\ndecoded %d tokens in %.2f s, speed: %.2f t/s (%d iterations)\n",
            n_decode_total, t, n_decode_total / t, iter);

    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}