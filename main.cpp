// get input prompt from user ✅
// send prompt from user to server ✅
// recieve prompt ✅
// tokenize prompt ✅
// send tokens to model ✅
// prefill ✅
// decode ✅
// stream output to user 
#include <iostream>    // Required for std::cout, std::cin, std::cerr
#include <string>
#include <vector>      // Required for std::string, std::getline
#include <cstring>     // Required for memset (if needed)
#include <sys/socket.h>// Required for socket(), bind(), listen(), accept(), recv(), send()
#include <arpa/inet.h> // Required for sockaddr_in, htons(), inet_pton()
#include <unistd.h>    // Required for close()
#include <stdio.h> 
#include <stdexcept>
#include <llama.h>
#include <chrono>
#include "common.h"

const int PORT = 8080;


// std::string get_prompt() { 
//     std::string prompt;
//     std::cout << "Enter your prompt for inference: \n> ";
//     if (std::getline(std::cin, prompt)) {
//         if (prompt.empty()) {
//             std::cerr << "Error: Prompt cannot be empty!" << std::endl;
//             return ""; // Return an empty string on error instead of '1'
//         }
//         std::cout << "Received.\n"; // Added missing semicolon
//     }
//     return prompt; 
// }

// int server() {
//     int server_fd, new_socket;
//     struct sockaddr_in address;
//     char buffer[1024] = {0};
//     socklen_t addrlen = sizeof(address);
    
//     address.sin_family = AF_INET;
//     address.sin_addr.s_addr = INADDR_ANY;
//     address.sin_port = htons(PORT);
    
//     server_fd = socket(AF_INET, SOCK_STREAM, 0);
//     if (server_fd < 0) {
//         perror("socket");
//         return -1;
//     }
    
//     // 2. FIXED: Allow instant port reuse to stop "Address already in use" errors
//     int opt = 1;
//     if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
//         perror("setsockopt(SO_REUSEADDR) failed");
//         close(server_fd);
//         return -1;
//     }
    
//     if (bind(server_fd, (struct sockaddr*)&address, addrlen) < 0) {
//         perror("bind");
//         close(server_fd);
//         return -1;
//     }
    
//     if (listen(server_fd, 3) < 0) {
//         perror("listen");
//         close(server_fd);
//         return -1;
//     }
    
//     std::cout << "Server listening on port " << PORT << ", waiting...\n";
//     new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen);
//     if (new_socket < 0) {
//         perror("accept");
//         close(server_fd);
//         return -1;
//     }
    
//     ssize_t n = recv(new_socket, buffer, sizeof(buffer) - 1, 0);
//     if (n > 0) {
//         llmInference->startCompletion(buffer);
//         std::string predictionToken;
//         while ((predictionToken = llmInference->completionLoop()) != "[EOS]") {
//             std::cout << predictionToken;
//             fflush(stdout);
//         }
//         std::cout << "\n";
//         std::cout << "Received: " << buffer << "\n";
//     }
    
//     close(new_socket);
//     close(server_fd);
//     return 0;
// }

// int client(const std::string& message) {
//     int client_fd;
//     struct sockaddr_in serv_addr;
    
//     serv_addr.sin_family = AF_INET;
//     serv_addr.sin_port = htons(PORT);
    
//     if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
//         perror("inet_pton");
//         return -1;
//     }
    
//     client_fd = socket(AF_INET, SOCK_STREAM, 0);
//     if (client_fd < 0) {
//         perror("socket");
//         return -1;
//     }
    
//     if (connect(client_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
//         perror("connect");
//         close(client_fd);
//         return -1;
//     }
    
//     send(client_fd, message.c_str(), message.length(), 0);
//     close(client_fd);
//     return 0;
// }

class LLMInference {

    llama_model* _model;
    llama_sampler* _sampler;
    llama_context* _ctx;
    llama_batch _batch;
    llama_token _currToken;

    std::vector<llama_chat_message> _messages;      //container to store the user/assistant messages
    std::vector<char> _formattedMessages;
    std::vector<llama_token> _promptTokens; 
    int _prevLen = 0;

    std::string _response = "";

    public:
        std::vector<llama_token> tokenizer(std::string prompt);
        void loadModel(const std::string& modelPath, float minP, float temperature);
        void addChatMessage(const std::string& message, const std::string& role);
        void startCompletion(const std::string& prompt);
        std::string completionLoop();
        void stopCompletion();

};

void LLMInference::loadModel(const std::string& model_path, float min_p, float temperature){
    // Create instance of llama_model
    llama_model_params model_params = llama_model_default_params();
    _model = llama_model_load_from_file(model_path.c_str(), model_params);

    if(!_model){
        throw std::runtime_error("load_model() failed");
    }

    // create an instance of llama_context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.no_perf = true;
    _ctx = llama_init_from_model(_model, ctx_params);

    if(!_ctx){
        throw std::runtime_error("llama_init_from_model() returned null");
    }

    // initialize sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    _sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(_sampler, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
    _messages.clear();
}

std::vector<llama_token> LLMInference::tokenizer(std::string prompt)
{
    bool add_bos = true;
    bool parse_special = true; 

    //get the prompt
    //call the tokenizer
    std::vector<llama_token> tokens = common_tokenize(_ctx, prompt, add_bos, parse_special);
    return tokens;

}



// void single_thread_perprocess(){}
// void pool_of_threads(){}
// void scheduler_static(){}
// void scheduler_dynamic(){}
// void inference_naiveKV(){}
// void inference_pagedKV(){}



void LLMInference::startCompletion(const std::string& prompt){
    // Decode, Sample, KV cache
    _promptTokens = common_tokenize(_ctx, prompt, true, true);

    // create a llama batch with a single sequence
    // see llama_batch_init for more details
    memset(&_batch, 0, sizeof(_batch));
    _batch.token = _promptTokens.data();
    _batch.n_tokens = static_cast<int32_t>(_promptTokens.size());
}

std::string LLMInference::completionLoop(){
    // check if the input length exceeds the maximum context size
    uint32_t contextSize = llama_n_ctx(_ctx);
    if (static_cast<uint32_t>(_batch.n_tokens) > contextSize) {
        std::cerr << "Context size exceeded" << '\n';
        exit(0);
    }

    // run the model
    if(llama_decode(_ctx, _batch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    // check if sampled token is EOG and convert the integer token to its
    // corresponding word-piece
    _currToken = llama_sampler_sample(_sampler, _ctx, -1);
    if (llama_vocab_is_eog(llama_model_get_vocab(_model), _currToken)) {
        _response.clear();
        return "[EOG]";
    }
    std::string piece = common_token_to_piece(_ctx, _currToken, true);
    // re-init batch with new token; K/V pairs of all prev tokens are cached
    _batch.token = &_currToken;
    _batch.n_tokens = 1;

    return piece;
}

// void LLMInfernce::stopCompletion(){

// }
// // Writing a deconstructor???

// LLMInference::~LLMInference() {
//     //free memory held by the message text in messages
//     //as strdup() created a malloc'ed copy)

//     for(llama_chat_message &message: _messages) {
//         delete message.content;
//     }

//     llama_kv_cache_clear(_ctx);
//     llama_sampler_fee(_sampler);
//     llama_free(_ctx);
//     llama_free_model(_model);
// }


// int main(int argc, char* argv[]) {

//     // LLM Setup

//     std::string modelPath = ""
//     float temperature = ;
//     float minP = ;
//     std::unique_ptr<LLM_Inference> llmInferece = std::make_unique<LLMInfference>();
//     llmInference->loadModel(modelPath, minP, temperature);

//     // llm LOGIC goes here 
//     llmInference->tokenizer()


//     while (true){

//     // Server client logic 
//         if (argc > 1 && std::string(argv[1]) == "server") {
//         server();
//     } 
//     else if (argc > 1 && std::string(argv[1]) == "client") {
//         std::string prompt = get_prompt(); 
//         if (!prompt.empty()) {
//             client(prompt); 
//         }
//     } 

//     return 0;
//     }

// }

int main (int argc, char* argv[]){
    
    int tokensGenerated = 0;
    std::string fullResponse;
    std::string prompt = "The capital of France is";

    LLMInference inference;

    inference.loadModel("/workspaces/llm-inference-server/models/gemma-3-1b-it-Q4_K_M.gguf", 0.05f, 0.8f);

    std::vector<llama_token> tokens = inference.tokenizer("The capital of France is");
    std::cout << "Token count: " << tokens.size() << std::endl;

    while (tokensGenerated < max_tokens) {
        if (tokensGenerated == 0) {
            auto t_start = std::chrono::steady_clock::now();

            inference.startCompletion(prompt);
            std::string firstPiece = inference.completionLoop(); //Prefill call

            auto t_first_token = std::chrono::steady_clock::now();
            auto ttft = std::chrono::duration_cast<std::chrono::milliseconds>(t_first_token - t_start);

            std::cout << "TTFT: " << ttft.count() << " ms" << std::endl;
        } else {

            //Decode loop
            std::string nextPiece = inference.completionLoop();
            if (nextPiece == "[EOG]") {
                std::cout << "End of Generation reached." << std::endl;
                break;
            }
            fullResponse += nextPiece;
            tokensGenerated++;
        }
    }

    return 0; 

}

