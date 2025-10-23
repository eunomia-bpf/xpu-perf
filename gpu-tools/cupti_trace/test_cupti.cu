#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>

// =============================================================================
// Configuration - Scaled up for ~1GB total memory usage
// =============================================================================
#define BATCH_SIZE 8
#define SEQ_LENGTH 512
#define HIDDEN_DIM 1024
#define NUM_HEADS 8
#define HEAD_DIM (HIDDEN_DIM / NUM_HEADS)
#define FFN_DIM (HIDDEN_DIM * 4)
#define NUM_LAYERS 3
#define VOCAB_SIZE 2000
// Main buffer size: 8 * 512 * 1024 * 4 = 16MB per buffer
// TransformerLayer allocations per layer: ~100MB
// 3 layers = ~300MB, plus I/O buffers = total ~400-500MB

// =============================================================================
// GPU Kernels - Transformer Operations
// =============================================================================

// Attention: Q * K^T
__global__ void attentionQKT(float *Q, float *K, float *scores,
                              int batch, int seq_len, int head_dim) {
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query position
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key position

    if (b < batch && i < seq_len && j < seq_len) {
        float sum = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            int q_idx = b * seq_len * head_dim + i * head_dim + k;
            int k_idx = b * seq_len * head_dim + j * head_dim + k;
            sum += Q[q_idx] * K[k_idx];
        }
        scores[b * seq_len * seq_len + i * seq_len + j] = sum / sqrtf((float)head_dim);
    }
}

// Softmax operation
__global__ void softmax(float *input, float *output, int batch, int seq_len) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && i < seq_len) {
        float max_val = -INFINITY;
        for (int j = 0; j < seq_len; j++) {
            int idx = b * seq_len * seq_len + i * seq_len + j;
            max_val = fmaxf(max_val, input[idx]);
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            int idx = b * seq_len * seq_len + i * seq_len + j;
            output[idx] = expf(input[idx] - max_val);
            sum += output[idx];
        }

        for (int j = 0; j < seq_len; j++) {
            int idx = b * seq_len * seq_len + i * seq_len + j;
            output[idx] /= sum;
        }
    }
}

// Attention scores * V
__global__ void attentionScoresV(float *scores, float *V, float *output,
                                  int batch, int seq_len, int head_dim) {
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && i < seq_len && k < head_dim) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            int score_idx = b * seq_len * seq_len + i * seq_len + j;
            int v_idx = b * seq_len * head_dim + j * head_dim + k;
            sum += scores[score_idx] * V[v_idx];
        }
        output[b * seq_len * head_dim + i * head_dim + k] = sum;
    }
}

// Matrix multiplication for FFN
__global__ void matmulFFN(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// GELU activation
__global__ void gelu(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Layer normalization
__global__ void layerNorm(float *input, float *output, float *gamma, float *beta,
                          int batch, int seq_len, int hidden_dim) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && i < seq_len) {
        float mean = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            mean += input[b * seq_len * hidden_dim + i * hidden_dim + j];
        }
        mean /= hidden_dim;

        float variance = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            float diff = input[b * seq_len * hidden_dim + i * hidden_dim + j] - mean;
            variance += diff * diff;
        }
        variance /= hidden_dim;

        float std = sqrtf(variance + 1e-5f);
        for (int j = 0; j < hidden_dim; j++) {
            int idx = b * seq_len * hidden_dim + i * hidden_dim + j;
            output[idx] = gamma[j] * (input[idx] - mean) / std + beta[j];
        }
    }
}

// Element-wise add (residual connection)
__global__ void residualAdd(float *input, float *residual, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + residual[idx];
    }
}

// =============================================================================
// CPU Processing Functions
// =============================================================================

class TokenEmbedding {
public:
    float *embeddings;
    int vocab_size;
    int embedding_dim;

    TokenEmbedding(int vocab, int dim) : vocab_size(vocab), embedding_dim(dim) {
        embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
        // Initialize with random values
        for (int i = 0; i < vocab_size * embedding_dim; i++) {
            embeddings[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    ~TokenEmbedding() {
        free(embeddings);
    }

    void embed(int *tokens, float *output, int num_tokens) {
        for (int i = 0; i < num_tokens; i++) {
            int token_id = tokens[i] % vocab_size;
            memcpy(output + i * embedding_dim,
                   embeddings + token_id * embedding_dim,
                   embedding_dim * sizeof(float));
        }
    }
};

class TextProcessor {
public:
    static void tokenize(const char *text, int *tokens, int max_tokens) {
        int len = strlen(text);
        for (int i = 0; i < max_tokens && i < len; i++) {
            tokens[i] = (int)text[i];  // Simple char-level tokenization
        }
    }

    static void detokenize(int *tokens, char *text, int num_tokens) {
        for (int i = 0; i < num_tokens; i++) {
            text[i] = (char)(tokens[i] % 128);
        }
        text[num_tokens] = '\0';
    }
};

// =============================================================================
// Network I/O Simulation
// =============================================================================

class NetworkSimulator {
private:
    int sock_fd;
    struct sockaddr_in server_addr;
    bool connected;

public:
    NetworkSimulator() : sock_fd(-1), connected(false) {}

    ~NetworkSimulator() {
        disconnect();
    }

    bool connect_to_server(const char *ip, int port) {
        sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_fd < 0) {
            printf("Warning: Could not create socket for network simulation\n");
            return false;
        }

        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        server_addr.sin_addr.s_addr = inet_addr(ip);

        // Try to connect (will fail if no server, that's ok for simulation)
        if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(sock_fd);
            sock_fd = -1;
            return false;
        }

        connected = true;
        return true;
    }

    void disconnect() {
        if (sock_fd >= 0) {
            close(sock_fd);
            sock_fd = -1;
        }
        connected = false;
    }

    bool send_request(const char *data, size_t len) {
        if (!connected) return false;
        ssize_t sent = send(sock_fd, data, len, MSG_NOSIGNAL);
        return sent > 0;
    }

    bool receive_response(char *buffer, size_t max_len) {
        if (!connected) return false;
        ssize_t received = recv(sock_fd, buffer, max_len, MSG_DONTWAIT);
        return received > 0;
    }

    // Simulate network activity
    void simulate_activity(const char *prompt, int iteration) {
        // Simulate sending prompt
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "Iteration %d: %s", iteration, prompt);
        usleep(1000);  // Simulate network latency

        // Simulate receiving response
        usleep(2000);
    }
};

// =============================================================================
// File I/O Operations - Simulating Prompt Prefix Cache
// =============================================================================

class PromptPrefixCache {
private:
    static const int MAX_CACHE_ENTRIES = 100;
    char cache_dir[256];
    char cache_files[MAX_CACHE_ENTRIES][256];
    int num_files;

public:
    PromptPrefixCache() : num_files(0) {
        snprintf(cache_dir, sizeof(cache_dir), "/tmp/llm_prefix_cache_%d", getpid());

        // Create cache directory
        char mkdir_cmd[512];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", cache_dir);
        system(mkdir_cmd);
    }

    ~PromptPrefixCache() {
        cleanup();
    }

    // Cache prompt prefix (KV cache simulation)
    void cache_prefix(const char *prompt_hash, float *kv_cache, size_t size, int iteration) {
        if (num_files >= MAX_CACHE_ENTRIES) return;

        char filename[512];
        snprintf(filename, sizeof(filename), "%s/prefix_%s_iter%d.cache",
                cache_dir, prompt_hash, iteration);

        FILE *fp = fopen(filename, "wb");
        if (fp) {
            // Write metadata
            fprintf(fp, "PROMPT_CACHE_V1\n");
            fprintf(fp, "SIZE=%zu\n", size);
            fprintf(fp, "ITERATION=%d\n", iteration);

            // Write cache data
            fwrite(kv_cache, sizeof(float), size, fp);
            fclose(fp);

            strncpy(cache_files[num_files], filename, sizeof(cache_files[0]) - 1);
            num_files++;
        }
    }

    // Load cached prefix
    bool load_prefix(const char *prompt_hash, float *kv_cache, size_t size, int iteration) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/prefix_%s_iter%d.cache",
                cache_dir, prompt_hash, iteration);

        FILE *fp = fopen(filename, "rb");
        if (fp) {
            char line[256];
            // Skip metadata lines
            fgets(line, sizeof(line), fp);
            fgets(line, sizeof(line), fp);
            fgets(line, sizeof(line), fp);

            size_t read = fread(kv_cache, sizeof(float), size, fp);
            fclose(fp);
            return read == size;
        }
        return false;
    }

    // Write inference log
    void write_log(const char *filename, int iteration, float *output, int num_tokens) {
        if (num_files >= MAX_CACHE_ENTRIES) return;

        char full_path[512];
        snprintf(full_path, sizeof(full_path), "%s/%s", cache_dir, filename);

        FILE *fp = fopen(full_path, "a");
        if (fp) {
            fprintf(fp, "Iteration %d: ", iteration);
            for (int i = 0; i < num_tokens && i < 10; i++) {
                fprintf(fp, "%.4f ", output[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);

            strncpy(cache_files[num_files], full_path, sizeof(cache_files[0]) - 1);
            num_files++;
        }
    }

    // Evict old cache entries (LRU simulation)
    void evict_old_entries(int max_age) {
        for (int i = 0; i < num_files; i++) {
            struct stat st;
            if (stat(cache_files[i], &st) == 0) {
                time_t now = time(NULL);
                if (difftime(now, st.st_mtime) > max_age) {
                    remove(cache_files[i]);
                }
            }
        }
    }

    // Cleanup all cache files
    void cleanup() {
        // Remove all tracked files
        for (int i = 0; i < num_files; i++) {
            remove(cache_files[i]);
        }

        // Remove cache directory
        char rmdir_cmd[512];
        snprintf(rmdir_cmd, sizeof(rmdir_cmd), "rm -rf %s", cache_dir);
        system(rmdir_cmd);

        num_files = 0;
    }
};

// =============================================================================
// Transformer Layer
// =============================================================================

class TransformerLayer {
public:
    // Device pointers
    float *d_Q, *d_K, *d_V;
    float *d_attn_scores, *d_attn_probs, *d_attn_output;
    float *d_ffn_intermediate, *d_ffn_output;
    float *d_ln_gamma, *d_ln_beta;
    float *d_residual;

    // Host pointers
    float *h_gamma, *h_beta;

    cudaStream_t stream;

    TransformerLayer() {
        printf("[DEBUG] Creating TransformerLayer...\n");
        // Allocate device memory
        size_t qkv_size = BATCH_SIZE * SEQ_LENGTH * HEAD_DIM * sizeof(float);
        size_t scores_size = BATCH_SIZE * SEQ_LENGTH * SEQ_LENGTH * sizeof(float);
        size_t hidden_size = BATCH_SIZE * SEQ_LENGTH * HIDDEN_DIM * sizeof(float);
        size_t ffn_size = BATCH_SIZE * SEQ_LENGTH * FFN_DIM * sizeof(float);

        printf("[DEBUG] Allocating QKV (%.2f MB each)...\n", qkv_size / 1024.0 / 1024.0);
        cudaMalloc(&d_Q, qkv_size);
        cudaMalloc(&d_K, qkv_size);
        cudaMalloc(&d_V, qkv_size);
        printf("[DEBUG] Allocating attention scores (%.2f MB)...\n", scores_size / 1024.0 / 1024.0);
        cudaMalloc(&d_attn_scores, scores_size);
        cudaMalloc(&d_attn_probs, scores_size);
        cudaMalloc(&d_attn_output, qkv_size);
        printf("[DEBUG] Allocating FFN (%.2f MB)...\n", ffn_size / 1024.0 / 1024.0);
        cudaMalloc(&d_ffn_intermediate, ffn_size);
        cudaMalloc(&d_ffn_output, hidden_size);
        cudaMalloc(&d_ln_gamma, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_ln_beta, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_residual, hidden_size);

        // Initialize layer norm parameters
        h_gamma = (float*)malloc(HIDDEN_DIM * sizeof(float));
        h_beta = (float*)malloc(HIDDEN_DIM * sizeof(float));
        for (int i = 0; i < HIDDEN_DIM; i++) {
            h_gamma[i] = 1.0f;
            h_beta[i] = 0.0f;
        }
        cudaMemcpy(d_ln_gamma, h_gamma, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ln_beta, h_beta, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);

        cudaStreamCreate(&stream);
    }

    ~TransformerLayer() {
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_attn_scores);
        cudaFree(d_attn_probs);
        cudaFree(d_attn_output);
        cudaFree(d_ffn_intermediate);
        cudaFree(d_ffn_output);
        cudaFree(d_ln_gamma);
        cudaFree(d_ln_beta);
        cudaFree(d_residual);
        free(h_gamma);
        free(h_beta);
        cudaStreamDestroy(stream);
    }

    void forward(float *d_input, float *d_output) {
        // Self-attention
        dim3 attn_block(16, 16);
        dim3 attn_grid((SEQ_LENGTH + 15) / 16, (SEQ_LENGTH + 15) / 16, BATCH_SIZE);

        attentionQKT<<<attn_grid, attn_block, 0, stream>>>(
            d_Q, d_K, d_attn_scores, BATCH_SIZE, SEQ_LENGTH, HEAD_DIM);

        dim3 softmax_grid((SEQ_LENGTH + 255) / 256, BATCH_SIZE);
        softmax<<<softmax_grid, 256, 0, stream>>>(
            d_attn_scores, d_attn_probs, BATCH_SIZE, SEQ_LENGTH);

        attentionScoresV<<<attn_grid, attn_block, 0, stream>>>(
            d_attn_probs, d_V, d_attn_output, BATCH_SIZE, SEQ_LENGTH, HEAD_DIM);

        // Residual connection
        int total_elements = BATCH_SIZE * SEQ_LENGTH * HIDDEN_DIM;
        residualAdd<<<(total_elements + 255) / 256, 256, 0, stream>>>(
            d_attn_output, d_input, d_residual, total_elements);

        // Layer norm
        dim3 ln_grid((SEQ_LENGTH + 255) / 256, BATCH_SIZE);
        layerNorm<<<ln_grid, 256, 0, stream>>>(
            d_residual, d_output, d_ln_gamma, d_ln_beta,
            BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM);

        cudaStreamSynchronize(stream);
    }
};

// =============================================================================
// Main Inference Pipeline
// =============================================================================

class InferencePipeline {
private:
    TransformerLayer **layers;
    TokenEmbedding *embedding;
    NetworkSimulator *network;
    PromptPrefixCache *prefix_cache;

    float *d_input, *d_output;
    float *h_input, *h_output;
    float *h_kv_cache;  // For prefix caching

    int num_layers;

public:
    InferencePipeline(int layers) : num_layers(layers) {
        printf("[INIT] Creating InferencePipeline with %d layers...\n", layers);
        // Initialize components
        printf("[INIT] Creating TokenEmbedding (vocab=%d, dim=%d)...\n", VOCAB_SIZE, HIDDEN_DIM);
        embedding = new TokenEmbedding(VOCAB_SIZE, HIDDEN_DIM);
        printf("[INIT] Creating NetworkSimulator...\n");
        network = new NetworkSimulator();
        printf("[INIT] Creating PromptPrefixCache...\n");
        prefix_cache = new PromptPrefixCache();

        // Try to connect to a mock server (will fail gracefully)
        network->connect_to_server("127.0.0.1", 8080);

        // Allocate transformer layers
        this->layers = new TransformerLayer*[num_layers];
        for (int i = 0; i < num_layers; i++) {
            this->layers[i] = new TransformerLayer();
        }

        // Allocate input/output buffers
        size_t io_size = BATCH_SIZE * SEQ_LENGTH * HIDDEN_DIM * sizeof(float);
        cudaMalloc(&d_input, io_size);
        cudaMalloc(&d_output, io_size);
        h_input = (float*)malloc(io_size);
        h_output = (float*)malloc(io_size);
        h_kv_cache = (float*)malloc(io_size);
    }

    ~InferencePipeline() {
        delete embedding;
        delete network;
        delete prefix_cache;  // This will cleanup all cache files
        for (int i = 0; i < num_layers; i++) {
            delete layers[i];
        }
        delete[] layers;
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        free(h_kv_cache);
    }

    // Generate simple hash for prompt (for cache key)
    void generate_prompt_hash(const char *prompt, char *hash, size_t hash_len) {
        unsigned long h = 5381;
        int c;
        const char *str = prompt;

        while ((c = *str++))
            h = ((h << 5) + h) + c;

        snprintf(hash, hash_len, "%lx", h);
    }

    void run_inference(const char *prompt, int iteration) {
        // 1. Network I/O - Simulate receiving request
        network->simulate_activity(prompt, iteration);

        // 2. Generate prompt hash for caching
        char prompt_hash[32];
        generate_prompt_hash(prompt, prompt_hash, sizeof(prompt_hash));

        // 3. Try to load from prefix cache (every 3rd iteration reuses cache)
        bool cache_hit = false;
        if (iteration % 3 == 0 && iteration > 0) {
            cache_hit = prefix_cache->load_prefix(prompt_hash, h_kv_cache,
                                                  BATCH_SIZE * SEQ_LENGTH * HIDDEN_DIM,
                                                  iteration - 3);
        }

        // 4. CPU: Tokenization
        int tokens[SEQ_LENGTH];
        TextProcessor::tokenize(prompt, tokens, SEQ_LENGTH);

        // 5. CPU: Embedding lookup
        embedding->embed(tokens, h_input, SEQ_LENGTH);

        // 6. Transfer to GPU
        size_t io_size = BATCH_SIZE * SEQ_LENGTH * HIDDEN_DIM * sizeof(float);
        cudaMemcpy(d_input, h_input, io_size, cudaMemcpyHostToDevice);

        // 7. GPU: Forward pass through transformer layers
        for (int i = 0; i < num_layers; i++) {
            layers[i]->forward(d_input, d_output);
            // Swap buffers for next layer
            float *temp = d_input;
            d_input = d_output;
            d_output = temp;
        }

        // 8. Transfer back to CPU
        cudaMemcpy(h_output, d_input, io_size, cudaMemcpyDeviceToHost);

        // 9. Save to prefix cache (simulate KV cache storage)
        if (iteration % 2 == 0) {
            memcpy(h_kv_cache, h_output, io_size);
            prefix_cache->cache_prefix(prompt_hash, h_kv_cache,
                                      BATCH_SIZE * SEQ_LENGTH * HIDDEN_DIM,
                                      iteration);
        }

        // 10. CPU: Post-processing
        char output_text[256];
        TextProcessor::detokenize(tokens, output_text, 100);

        // 11. File I/O - Log results
        if (iteration % 5 == 0) {
            char log_filename[64];
            snprintf(log_filename, sizeof(log_filename), "inference_%d.log", iteration);
            prefix_cache->write_log(log_filename, iteration, h_output, SEQ_LENGTH);
        }

        // 12. Evict old cache entries (every 20 iterations)
        if (iteration % 20 == 0) {
            prefix_cache->evict_old_entries(60);  // Evict entries older than 60 seconds
        }

        // 13. Network I/O - Simulate sending response
        network->simulate_activity(output_text, iteration);
    }
};

// =============================================================================
// Global cleanup and signal handling
// =============================================================================

InferencePipeline *g_pipeline = NULL;
volatile sig_atomic_t g_interrupted = 0;

void signal_handler(int signum) {
    printf("\n\n[SIGNAL] Received signal %d, cleaning up...\n", signum);
    g_interrupted = 1;

    // Cleanup pipeline (will clean all cache files)
    if (g_pipeline) {
        printf("[CLEANUP] Destroying inference pipeline and cache files...\n");
        delete g_pipeline;
        g_pipeline = NULL;
    }

    printf("[CLEANUP] Complete. Exiting.\n");
    exit(signum);
}

void setup_signal_handlers() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGINT, &sa, NULL);   // Ctrl+C
    sigaction(SIGTERM, &sa, NULL);  // Termination signal
    sigaction(SIGHUP, &sa, NULL);   // Hangup
}

// =============================================================================
// Main Function
// =============================================================================

int main() {
    printf("=============================================================\n");
    printf("LLM Inference Simulator with GPU/CPU/Network/IO Operations\n");
    printf("=============================================================\n");
    printf("Configuration:\n");
    printf("  - Batch Size: %d\n", BATCH_SIZE);
    printf("  - Sequence Length: %d\n", SEQ_LENGTH);
    printf("  - Hidden Dimension: %d\n", HIDDEN_DIM);
    printf("  - Number of Layers: %d\n", NUM_LAYERS);
    printf("  - Number of Attention Heads: %d\n", NUM_HEADS);
    printf("  - Duration: 10 seconds\n");
    printf("  - Prefix Cache: Enabled (simulating KV cache)\n");
    printf("=============================================================\n\n");

    // Initialize CUDA device first
    printf("[INIT] Initializing CUDA device...\n");
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Force CUDA context creation with a dummy operation
    float *dummy;
    err = cudaMalloc(&dummy, sizeof(float));
    if (err != cudaSuccess) {
        printf("[ERROR] Failed initial cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaFree(dummy);
    printf("[INIT] CUDA device initialized successfully\n\n");

    // Setup signal handlers for cleanup
    setup_signal_handlers();

    // Initialize random seed
    srand(time(NULL));

    // Create inference pipeline
    printf("[INIT] Initializing inference pipeline...\n");
    g_pipeline = new InferencePipeline(NUM_LAYERS);
    printf("[INIT] Cache directory created: /tmp/llm_prefix_cache_%d\n", getpid());

    // Test prompts
    const char *prompts[] = {
        "What is the meaning of artificial intelligence?",
        "Explain how transformers work in deep learning",
        "Generate a summary of recent advances in AI",
        "Describe the impact of large language models",
        "What are the ethical considerations in AI development?"
    };
    int num_prompts = 5;

    // Main inference loop
    printf("Starting inference loop (will run for ~30 seconds)...\n");
    printf("Press Ctrl+C to stop and cleanup...\n\n");
    time_t start_time = time(NULL);
    int iteration = 0;

    while (difftime(time(NULL), start_time) < 10.0 && !g_interrupted) {
        iteration++;

        // Select a prompt
        const char *prompt = prompts[iteration % num_prompts];

        // Run inference
        g_pipeline->run_inference(prompt, iteration);

        // Progress reporting
        if (iteration % 5 == 0) {
            double elapsed = difftime(time(NULL), start_time);
            printf("[%.1fs] Completed %d inference iterations (cache activity ongoing)\n",
                   elapsed, iteration);
        }

        // Small delay to allow profiler sampling
        usleep(50000);  // 50ms between iterations
    }

    // Cleanup
    printf("\n=============================================================\n");
    if (g_interrupted) {
        printf("Inference interrupted by user after %d iterations\n", iteration);
    } else {
        printf("Inference completed: %d iterations in 10 seconds\n", iteration);
        printf("Average throughput: %.2f iterations/second\n", iteration / 10.0);
    }
    printf("=============================================================\n");

    printf("[CLEANUP] Cleaning up pipeline and cache files...\n");
    delete g_pipeline;
    g_pipeline = NULL;
    printf("[CLEANUP] All cache files removed successfully.\n");

    return 0;
}
