#include "llama.h"
#include "common.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std:: string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, true, true);

    std::vector<llama_token> tokens(n);

    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0 ) {
        fprintf(stderr, "tokenize: failed\n");
        exit(1);
    }

    return tokens;
}

void tokens_to_text(const llama_vocab * vocab, const std::vector<llama_token> & prompt_tokens){
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            exit(1);
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }
}

struct Sequence {
    int seq_id;
    std::string prompt;
    std::vector<llama_token> tokens;
    int n_past = 0;
    int batch_idx = -1;
    int n_decode = 0;
    bool done = false;
    std::string output; 
};

std::string detokenize(const llama_vocab * vocab, llama_token id) {
    char buf[128];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) { fprintf(stderr, "detokenize: failed\n"); exit(1); }
    return std::string(buf, n);
}


int main(int argc, char* argv[]){
    // set up model weights + tokenizer
    std::string model_path = "models/gemma-3-1b-it-Q4_K_M.gguf";
    std::vector<std::string> prompts{
        "The capital of france is",
        "The capital of India",
        "The capital of Spain is"
    };

    std::vector<Sequence> seqs;

    int ngl = 99;
    int n_predict = 5; //no of tokens to predict


    ggml_backend_load_all();

    // initialize model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    for (int i = 0; i < prompts.size(); i++) {
        Sequence s;
        s.seq_id = i;
        s.prompt = prompts[i];
        s.tokens = tokenize(vocab, prompts[i]);
        seqs.push_back(s);
    }

    int n_seq_max = seqs.size();
    
    int n_prompt_total = 0;
    int n_longest = 0;
    for (auto & s : seqs) {
        n_prompt_total += s.tokens.size();
        n_longest = std::max(n_longest, (int) s.tokens.size());
    }

    int n_max_tokens = n_prompt_total; 

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_seq_max * (n_longest + n_predict);
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_max_tokens;
    // enable performance counters
    ctx_params.n_seq_max = n_seq_max;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token : Tokeniser test make it into a function

    for (int i = 0; i < prompts.size(); i++ ) {
        tokens_to_text(vocab, seqs[i].tokens);
    }

    // Prefill
    llama_batch batch = llama_batch_init(n_max_tokens, 0, n_seq_max);

    common_batch_clear(batch);

    for (auto & s : seqs) {
    for (int i = 0; i < s.tokens.size(); i++) {
        bool is_last = (i == s.tokens.size() - 1);
        common_batch_add(batch, s.tokens[i], i, { s.seq_id }, is_last);
        if (is_last) s.batch_idx = batch.n_tokens - 1;
    }
    s.n_past = s.tokens.size();
}

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prefill failed\n");
        return 1;
    }

    // Decode loop 
    const auto t_main_start = ggml_time_us();

    while(true){
        // looping through the sequence
        for (auto & s : seqs) {
            if (s.done) continue;

            llama_token tok = llama_sampler_sample(smpl, ctx, s.batch_idx);

            //Check if EOG
            if (llama_vocab_is_eog(vocab, tok) || s.n_decode >= n_predict){
                s.done = true;
                continue;
            }

            s.output += detokenize(vocab, tok);
            s.tokens.push_back(tok);
            s.n_decode++;
        }

        bool any_running = false;
        for (auto & s : seqs) if (!s.done) any_running = true;
        if (!any_running) break;

        // Rebuild the batch: one token per unfinished sequeence
        common_batch_clear(batch);
        for (auto & s : seqs) {
            if (s.done) continue;
            common_batch_add(batch, s.tokens.back(), s.n_past, { s.seq_id }, true);
            s.batch_idx = batch.n_tokens - 1;
            s.n_past++;
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        
    }

    const auto t_main_end = ggml_time_us();

    // results
    printf("\n");
    for (auto & s : seqs) {
        printf("[seq %d] %s -->%s\n", s.seq_id, s.prompt.c_str(), s.output.c_str());
    }
    int n_decode_total = 0;
    for (auto & s : seqs) n_decode_total += s.n_decode;

    const float t = (t_main_end - t_main_start) / 1000000.0f;
    fprintf(stderr, "\ndecoded %d tokens in %.2f s, speed: %.2f t/s\n",
            n_decode_total, t, n_decode_total / t);

    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
    }


