// get input prompt from user ✅
// send prompt from user to server ✅
// recieve prompt ✅
// tokenize prompt
// send tokens to model 
// run inference
// decode output
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

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <thread>
#include <stdlib.h>

#include <httplib.h>
#include <nlohmann/json.hpp>   // add to CMakeLists

const int PORT = 8080;

struct Request{
    uint64_t id;
    std::vector<llama_token> tokens;
    int client_fd; // file descriptor for the client socket
}; 

class RequestQueue{
    std::queue<Request> _queue;
    std::mutex _mutex;
    std::condition_variable _cv;

    public:
        void push(Request req){
            {
            std::lock_guard<std::mutex> lock(_mutex);
            _queue.push(std::move(req));
        }
        _cv.notify_one();
        }


        Request pop(){
            std::unique_lock<std::mutex> lock(_mutex);
            _cv.wait(lock, [this]{ return !_queue.empty(); });
            Request req = std::move(_queue.front());
            _queue.pop();
            return req;
        }
};

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
        ~LLMInference();

};

class ResultChannel{

    public:
        void push()
        void pop()
}

class Scheduler(
    
)

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

void LLMInference::stopCompletion(){

}
// Writing a deconstructor???

LLMInference::~LLMInference() {
    //free memory held by the message text in messages
    //as strdup() created a malloc'ed copy)

    for(llama_chat_message &message: _messages) {
        delete message.content;
    }

    llama_kv_cache_clear(_ctx);
    llama_sampler_free(_sampler);
    llama_free(_ctx);
    llama_model_free(_model);
}

std::string get_prompt() { 
    std::string prompt;
    std::cout << "Enter your prompt for inference: \n> ";
    if (std::getline(std::cin, prompt)) {
        if (prompt.empty()) {
            std::cerr << "Error: Prompt cannot be empty!" << std::endl;
            return ""; // Return an empty string on error instead of '1'
        }
        std::cout << "Received.\n"; // Added missing semicolon
    }
    return prompt; 
}

int create_listening_socket(){
    // Set up server
    int server_fd, new_socket;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);    
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0){
        perror("socket");
        return -1;
    }

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(SO_REUSEADDR) failed");
        close(server_fd);
        return -1;
    }

    if (bind(server_fd, (struct sockaddr*)&address, addrlen) < 0){
        perror("bind");
        close(server_fd);
        return -1;
    } 

    if (listen(server_fd, 3) < 0){
        perror("listen");
        close(server_fd);
        return -1;
    }

    return server_fd;

}

void handle_connection(int client_fd, LLMInference& inference, RequestQueue& queue, std::atomic<uint64_t>& nextRequestId){
    
    // 2. tokenize it 
    // 3. build a Request (tokens + a way to send results back to THIS connection)
    // 4. queue.push(request)
    // 5. loo: wait for next request -> send to client_fd 

    // 1. recv the prompt
    char buffer[1024] = {0};
    ssize_t n = recv(client_fd, buffer, sizeof(buffer)-1, 0);
    if (n <= 0){
        perror("recv");
        close(client_fd);
        return;
    }

    std::string prompt(buffer, n);

    //2. Tokenize
    std::vector<llama_token> tokens = inference.tokenizer(prompt);
    
    // 3. Build a Request
    Request req;
    req.id = nextRequestId.fetch_add(1);
    req.tokens = tokens;
    req.client_fd = client_fd;

    // 4. hand off to scheduler
    queue.push(req);

    //test
    std::cout << "[enqueued] id=" << req.id 
          << " tokens=" << req.tokens.size() 
          << " fd=" << req.client_fd << std::endl;

    // 5. wait for generated pieces an dstream them to the client_fd

    // 6. close (client_fd) once generation ends 

    }

void run_thread_per_connection(int server_fd, LLMInference& inference, RequestQueue& queue, std::atomic<uint64_t>& nextRequestId){
    // one thread is launched for every incoming request
    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0){
            perror("accept");
            continue;
        }

        // Connection handed to its own thread and go back to accepting
        std::thread t(handle_connection, client_fd, std::ref(inference), std::ref(queue), std::ref(nextRequestId));
        t.detach();

    }
}

void thread_loop(){
    // a pool of threads are launched at once and when the limit is hit the request is enqueued
}

void event_loop(){

}


int client(const std::string& message) {
    int client_fd;
    struct sockaddr_in serv_addr;
    
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        perror("inet_pton");
        return -1;
    }
    
    client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd < 0) {
        perror("socket");
        return -1;
    }
    
    if (connect(client_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connect");
        close(client_fd);
        return -1;
    }
    
    send(client_fd, message.c_str(), message.length(), 0);
    close(client_fd);
    return 0;
}

// BUild these before main can run 



// add RequesTimings to Request 


int main (int argc, char* argv[]){
    if (argc<2) {
        std::cout <<" Usage: " << argv[0] << " [server|client|test]\n";
        return 1; 
    }

    std::string mode = argv[1]; 

    if (mode == "client"){
        std::string prompt = get_prompt();
        if (prompt.empty()) return 1;
        return client(prompt);
    }

    // Load model 

    LLMInference inference;
    inference.loadModel("/workspaces/llm-inference-server/models/gemma-3-1b-it-Q4_K_M.gguf", 0.05f, 0.8f);
    
    // Test mode : Single request TTFT 

    if (mode == "test"){
        std::string prompt = "The capital of France is";
        auto t_start = std::chrono::steady_clock::now();
        inference.startCompletion(prompt);
        std::string firstPiece = inference.completionLoop(); //Prefill call
        auto t_first_token = std::chrono::steady_clock::now();
        auto ttft = std::chrono::duration_cast<std::chrono::milliseconds>(t_first_token - t_start);
        std::cout << "First token piece: " << firstPiece << std::endl;
        std::cout << "TTFT: " << ttft.count() << " ms" << std::endl;
        
        return 0; 

    }

    // Shared states owned by main

    RequestQueue queue;
    std::atomic<uint64_t> nextRequestId{0};
    MetricsCollector metrics; 

    // Scheduler thread
    std::thread schedulerThread(run_scheduler, 
        std::ref(inference), 
        std::ref(inference), 
        std::ref(queue), 
        std::ref(metrics));

    schedulerThread.detach();

    // HTTP layer
    httplib::Server svr;

    svr.Post("v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        auto arrived = std::chrono::steady_clock::now();

        // Parse the JSON request
        std::string prompt;
        int maxTokens = 128;
        try {
            auto body = nlohmann::json::parse(httpReq.body);
            prompt = body.at("prompt").get<std::string>();
            if (bodty.contains("max_tokens")) maxTokens = body["max_tokens"];
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("{\"error\": \"Invalid JSON or missing 'prompt' field.\"}",
            return;

        }

        // Tokenize safe from thr handler 
        auto tokens = inference.tokenizer(prompt);
        auto tokenized = std::chrono::steady_clock::now();
        
        // build request 
        auto channel = std::make_shared<ResultChannel>();

        Request req;
        req.id = nextRequestId.fetch_add(1);
        req.tokens = tokens;
        req.maxTokens = maxTokens;
        req.channel = channel;
        req.timinigs.arrived = arrived;
        req.timings.tokenized = tokenized;
        req.timings.enqueued = std::chrono::steady_clock::now();

        queue.push(std::move(req));

        // Streaming results back over SSE
        res.set_chunked_content_provider(
            "text/event-stream",
            [channel](size_t /*offset*/, httplib::DataSink& sink) {
                std::string piece = channel->pop();

                if piece == "[EOG]"){
                    const std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size()); 
                    sink.done();
                    return false;
                }

                std::string chunk = "data: " + piece + "\n\n";
                sink.write(chunk.data(), chunk.size());
                return true; // Continue streaming
            }
        );

}); 

    / --- metrics endpoint ---
    svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics.report(), "text/plain");
    });

    std::cout << "listening on 8080\n";
    svr.listen("0.0.0.0", 8080);

    return 0;
}
