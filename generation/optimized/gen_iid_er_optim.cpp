// underlying: Erdos Renyi
// binding: parallel binding

#include <omp.h>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <mutex>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

typedef boost::multiprecision::cpp_dec_float_50 float50;  // 50 digits of precision

void display_progress(int step, int total_steps, chrono::steady_clock::time_point start_time, int bar_width = 50) {
    auto current_time = chrono::steady_clock::now();
    chrono::duration<float> elapsed = current_time - start_time;
    float progress = static_cast<float>(step) / total_steps;
    int pos = static_cast<int>(bar_width * progress);

    // Calculate elapsed_time and remaining_time
    float elapsed_time = elapsed.count();
    float remaining_time = 0.0;
    
    // Avoid division by zero and unrealistic estimates
    if (progress > 0.01) { // Only calculate ETA when progress > 1%
        remaining_time = (elapsed_time / progress) - elapsed_time;
        // Cap remaining time to reasonable values
        if (remaining_time < 0 || remaining_time > 3600) { // Cap at 1 hour
            remaining_time = 0.0;
        }
    }

    cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            cout << "=";
        else if (i == pos)
            cout << ">";
        else
            cout << " ";
    }
    cout << "] " << fixed << setprecision(2) << progress * 100.0 << " %, Elapsed: "
         << step << "/" << total_steps << ", "
         << elapsed_time << " s";
    
    if (remaining_time > 0) {
        cout << ", ETA: " << remaining_time << " s";
    }
    
    cout << "\r";
    cout.flush();
}

// Helper function to encode pair (v1, v2) where v1 < v2 into unsigned long long
inline unsigned long long encode_pair(unsigned int v1, unsigned int v2, unsigned int n_nodes) {
    if (v1 > v2) swap(v1, v2);
    return static_cast<unsigned long long>(v1) * n_nodes + v2;
}

// Helper function to decode pair ID back to (v1, v2)
inline pair<unsigned int, unsigned int> decode_pair(unsigned long long pair_id, unsigned int n_nodes) {
    unsigned int v1 = static_cast<unsigned int>(pair_id / n_nodes);
    unsigned int v2 = static_cast<unsigned int>(pair_id % n_nodes);
    return make_pair(v1, v2);
}

// Memory tracking functions
struct MemoryInfo {
    size_t vm_peak_kb = 0;    // Peak virtual memory
    size_t vm_size_kb = 0;    // Current virtual memory
    size_t vm_rss_kb = 0;     // Current physical memory (RSS)
    size_t vm_hwm_kb = 0;     // Peak physical memory
};

MemoryInfo get_memory_usage() {
    MemoryInfo info;
    
    ifstream status_file("/proc/self/status");
    if (!status_file.is_open()) {
        return info; // Return zeros if can't read (non-Linux systems)
    }
    
    string line;
    while (getline(status_file, line)) {
        if (line.substr(0, 7) == "VmPeak:") {
            sscanf(line.c_str(), "VmPeak: %zu kB", &info.vm_peak_kb);
        } else if (line.substr(0, 7) == "VmSize:") {
            sscanf(line.c_str(), "VmSize: %zu kB", &info.vm_size_kb);
        } else if (line.substr(0, 6) == "VmRSS:") {
            sscanf(line.c_str(), "VmRSS: %zu kB", &info.vm_rss_kb);
        } else if (line.substr(0, 6) == "VmHWM:") {
            sscanf(line.c_str(), "VmHWM: %zu kB", &info.vm_hwm_kb);
        }
    }
    
    status_file.close();
    return info;
}

void print_memory_info(const string& label, const MemoryInfo& info) {
    cout << label << ":" << endl;
    cout << "  Current RSS: " << fixed << setprecision(2) << info.vm_rss_kb / 1024.0 << " MB" << endl;
    cout << "  Peak RSS: " << fixed << setprecision(2) << info.vm_hwm_kb / 1024.0 << " MB" << endl;
    cout << "  Current Virtual: " << fixed << setprecision(2) << info.vm_size_kb / 1024.0 << " MB" << endl;
    cout << "  Peak Virtual: " << fixed << setprecision(2) << info.vm_peak_kb / 1024.0 << " MB" << endl;
}

// input //
// n_nodes: number of nodes
// prob: the uniform edge probability in each round
// alpha: the uniform binding strength
// n_rounds: number of total rounds
// output //
// a list of edges
// vector<int> generate_edges(int n_nodes, double prob, double alpha, int n_rounds) {
vector<vector<unsigned int>> generate_edges(unsigned int n_nodes, float50 prob, float50 alpha, int n_rounds) {
    // initialize an empty edge list
    // vector<pair<int, int>> edges;
    vector<vector<unsigned int>> cliques; // Each element is a clique

    // in total "n_rounds" rounds
    // we first compute the number of rounds with insertion
    // then for each "success" round, we sample nodes in "nodes", where each node is sampled with probabilitiy "alpha"
    // then we add a clique between the sampled nodes
    
    vector<vector<vector<unsigned int>>> cliques_omp(omp_get_max_threads());
    atomic<int> progress(0);
    auto start_time = chrono::steady_clock::now();
    static mutex progress_mutex;

    // compute the number of rounds with non-empty insertion
    unsigned int n_rounds_gen;
    if (prob == 1.0) {
        n_rounds_gen = n_rounds;
    } else {
        // binomial distribution (n_rounds, prob)
        // random number generator with better seeding
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        mt19937 rng(seed);
        // case prob to double
        double prob_double = prob.convert_to<double>();
        binomial_distribution<int> dist_binom(n_rounds, prob_double);
        n_rounds_gen = dist_binom(rng);
    }

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < n_rounds_gen; ++i) {
        int tid = omp_get_thread_num();

        // Improved random number generator with thread-specific seeding
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + 
                       tid * 1000000 + i;  // Ensure unique seeds per thread and iteration
        mt19937 rng(seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        // sample nodes
        vector<unsigned int> sampled_nodes;
        
        // First, determine how many nodes to sample using binomial distribution
        double alpha_double = alpha.convert_to<double>();
        binomial_distribution<unsigned int> dist_binom_nodes(n_nodes, alpha_double);
        unsigned int n_nodes_to_sample = dist_binom_nodes(rng);
        
        if (n_nodes_to_sample >= 2) {
            // Use efficient sampling without creating full node vector
            sampled_nodes.reserve(n_nodes_to_sample);
            unordered_set<unsigned int> selected_nodes;
            selected_nodes.reserve(n_nodes_to_sample);
            
            if (n_nodes_to_sample <= n_nodes / 2) {
                // When sampling few nodes, use rejection sampling
                uniform_int_distribution<unsigned int> node_dist(0, n_nodes - 1);
                while (selected_nodes.size() < n_nodes_to_sample) {
                    unsigned int node = node_dist(rng);
                    selected_nodes.insert(node);
                }
            } else {
                // When sampling many nodes, sample the complement (nodes to exclude)
                int n_nodes_to_exclude = n_nodes - n_nodes_to_sample;
                unordered_set<unsigned int> excluded_nodes;
                excluded_nodes.reserve(n_nodes_to_exclude);
                uniform_int_distribution<unsigned int> node_dist(0, n_nodes - 1);
                while (excluded_nodes.size() < n_nodes_to_exclude) {
                    unsigned int node = node_dist(rng);
                    excluded_nodes.insert(node);
                }
                // Add all non-excluded nodes
                for (unsigned int j = 0; j < n_nodes; ++j) {
                    if (excluded_nodes.find(j) == excluded_nodes.end()) {
                        selected_nodes.insert(j);
                    }
                }
            }
            
            // Convert set to vector
            sampled_nodes.assign(selected_nodes.begin(), selected_nodes.end());
        }

        // if the number of sampled nodes is less than 2, skip this round
        if (sampled_nodes.size() < 2) {
            int current_progress = ++progress;
            // Fix race condition: any thread can update progress display periodically
            if (current_progress % max(1U, n_rounds_gen / 100U) == 0 || tid == 0) {
                lock_guard<mutex> lock(progress_mutex);
                display_progress(current_progress, n_rounds_gen, start_time);
            }
            continue;
        }

        int current_progress = ++progress;
        // Fix race condition: any thread can update progress display periodically  
        if (current_progress % max(1U, n_rounds_gen / 100U) == 0 || tid == 0) {
            lock_guard<mutex> lock(progress_mutex);
            display_progress(current_progress, n_rounds_gen, start_time);
        }        
        cliques_omp[tid].push_back(sampled_nodes);
    }
    
    // Ensure we display 100% completion
    display_progress(n_rounds_gen, n_rounds_gen, start_time);
    cout << endl;

    // merge the cliques
    for (const auto& c : cliques_omp) {
        cliques.insert(cliques.end(), c.begin(), c.end());
    }

    return cliques;
}

// main function
int main(int argc, char const* argv[]) {
    // read the arguments
    // argv[1]: input filename
    // argv[2]: output filename
    // argv[3]: number of generated graphs
    // argv[4]: up_scale
    // argv[5]: expand

    // read the file name from argv into a string
    if (argc != 6) {
        cout << "Usage: " << argv[0]
             << " <filename_in>"
             << " <filename_out>"
             << " <n_graphs>"
             << " <up_scale>"
             << " <expand>"
             << endl;
        return 1;
    }
    string filename_in = argv[1];
    string filename_out = argv[2];
    unsigned int n_graphs = atoi(argv[3]);
    unsigned int up_scale = atoi(argv[4]);
    bool expand = atoi(argv[5]);

    // Input validation
    if (n_graphs <= 0) {
        cerr << "Error: Number of graphs must be positive" << endl;
        return 1;
    }

    // read the input file: number of nodes, density, binding strength, number of rounds, probability in each round
    ifstream infile;
    infile.open(filename_in);
    if (!infile.is_open()) {
        cerr << "Error: Cannot open input file " << filename_in << endl;
        return 1;
    }
    
    unsigned int n_nodes;
    unsigned int n_rounds;
    float50 p, alpha, p_round;
    infile >> n_nodes >> p >> alpha >> n_rounds >> p_round;
    n_nodes = n_nodes * up_scale;
    infile.close();

    // Input validation
    if (n_nodes == 0) {
        cerr << "Error: Number of nodes must be positive" << endl;
        return 1;
    }
    if (n_rounds <= 0) {
        cerr << "Error: Number of rounds must be positive" << endl;
        return 1;
    }
    if (p < 0.0 || p > 1.0) {
        cerr << "Error: Density p must be between 0 and 1" << endl;
        return 1;
    }
    if (alpha < 0.0 || alpha > 1.0) {
        cerr << "Error: Binding strength alpha must be between 0 and 1" << endl;
        return 1;
    }

    // print the arguments
    cout << "Number of nodes: " << n_nodes << endl;
    cout << "Density: " << p << endl;
    cout << "Binding strength alpha: " << alpha << endl;
    cout << "Number of rounds: " << n_rounds << endl;
    cout << "Output file: " << filename_out << endl;
    cout << "Number of generated graphs: " << n_graphs << endl;
    cout << "Expand: " << (expand ? "True" : "False") << endl;

    // compute the probability of remaining edges
    float50 p_remain = 0.0;
    bool has_remain = false;
    if (p_round > 1.0) {
        p_round = 1.0;
        float50 p_max = 1.0 - pow(1.0 - alpha * alpha, n_rounds);
        p_remain = 1.0 - (1.0 - p) / (1.0 - p_max);
        has_remain = true;
    }
    cout << "p: " << p << endl;
    cout << "p_round: " << p_round << endl;
    cout << "p_remain: " << p_remain << endl;   

    // Memory tracking
    MemoryInfo initial_memory = get_memory_usage();
    print_memory_info("Initial memory usage", initial_memory);
    cout << endl;

    // Vector to collect generation and write times
    vector<double> generation_times;
    vector<double> write_times;
    vector<MemoryInfo> memory_snapshots;

    // generate n_graphs graphs
    for (auto i_graph = 0; i_graph < n_graphs; i_graph++) {
        // Periodic memory reporting for long runs
        if (i_graph > 0 && i_graph % max(1U, n_graphs / 10U) == 0) {
            MemoryInfo current_memory = get_memory_usage();
            cout << "Progress: " << i_graph << "/" << n_graphs 
                 << " graphs, Current RSS: " << fixed << setprecision(1)
                 << current_memory.vm_rss_kb / 1024.0 << " MB" << endl;
        }
        
        // generate edge using binding
        auto start = std::chrono::steady_clock::now();  // Start the timer

        // vector<pair<int, int>> edges = generate_edges(n_nodes, p_round, alpha, n_rounds);
        vector<vector<unsigned int>> cliques = generate_edges(n_nodes, p_round, alpha, n_rounds);

        // add the remaining edges if has_remain
        vector<unsigned long long> remaining_pairs;  // Changed to unsigned long long
        if (has_remain) {
            // First get the number of pairs we will add with probability p_remain
            unsigned long long n_total_pairs = static_cast<unsigned long long>(n_nodes) * (n_nodes - 1) / 2;
            unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + i_graph;
            mt19937 rng(seed);

            // Now we need to sample pairs, so that each pair is sampled with probability p_remain
            // To speed up this, we do independent sampling
            // In each round, we sample a pair uniformly at random
            // We now compute the number of rounds we need, so that each pair is sampled with probability p_remain
            // n_rounds_remain = log_{1 - 1/n_total_pairs} (1 - p_remain)

            float50 p_base = - 1.0 / n_total_pairs;
            float50 n_rounds_remain = log1p(-p_remain) / log1p(p_base);
            
            unsigned int n_rounds_remain_int = static_cast<unsigned int>(n_rounds_remain);
            
            // Use OpenMP to parallelize the sampling of pairs
            vector<vector<unsigned long long>> remaining_pairs_omp(omp_get_max_threads());
            
            #pragma omp parallel for schedule(dynamic, 32)
            for (int i_round = 0; i_round < n_rounds_remain_int; i_round++) {
                int tid = omp_get_thread_num();
                
                // Each thread needs its own RNG with a unique seed
                unsigned seed_thread = seed + i_round + tid * n_rounds_remain_int;
                mt19937 rng_thread(seed_thread);
                
                // sample a pair uniformly at random
                unsigned int node1, node2;
                do {
                    uniform_int_distribution<unsigned int> node_dist(0, n_nodes - 1);                
                    node1 = node_dist(rng_thread);
                    node2 = node_dist(rng_thread);
                } while (node1 == node2);
                if (node1 > node2) swap(node1, node2); // Ensure node1 < node2
                unsigned long long pair_id = encode_pair(node1, node2, n_nodes);
                remaining_pairs_omp[tid].push_back(pair_id);
            }
            
            // Collect results from all threads
            for (const auto& thread_pairs : remaining_pairs_omp) {
                remaining_pairs.insert(remaining_pairs.end(), thread_pairs.begin(), thread_pairs.end());
            }            
        }

        auto end = std::chrono::steady_clock::now();                                         // Stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  // Calculate the duration in milliseconds
        double generation_time_ms = duration.count();
        generation_times.push_back(generation_time_ms);

        // Track memory after generation
        MemoryInfo post_generation_memory = get_memory_usage();
        memory_snapshots.push_back(post_generation_memory);

        // Write the cliques and remaining edges directly (much more efficient!)
        ofstream outfile;
        outfile.open(filename_out + "_" + to_string(i_graph) + ".txt");       
        auto start_write = std::chrono::steady_clock::now();  // Start the timer        
        
        // Write cliques section
        outfile << "CLIQUES " << cliques.size() << endl;
        for (const auto& clique : cliques) {
            outfile << clique.size();
            for (unsigned int node : clique) {
                outfile << " " << node;
            }
            outfile << endl;
        }
        
        // Write remaining edges section
        outfile << "REMAINING_EDGES " << remaining_pairs.size() << endl;
        for (unsigned long long pair_id : remaining_pairs) {
            auto edge = decode_pair(pair_id, n_nodes);
            outfile << edge.first << " " << edge.second << endl;
        }

        auto end_write = std::chrono::steady_clock::now();                                                     // Stop the timer
        auto duration_write = std::chrono::duration_cast<std::chrono::milliseconds>(end_write - start_write);  // Calculate the duration in milliseconds
        double write_time_ms = duration_write.count();
        write_times.push_back(write_time_ms);

        outfile.close();

        if (expand) {
            ofstream outfile;
            outfile.open(filename_out + "_" + to_string(i_graph) + "_expand.txt");
            unordered_set<unsigned long long> generated_edges;
            for (const auto& clique : cliques) {
                auto size_clique = clique.size();
                for (unsigned int i = 0; i < size_clique; ++i) {
                    auto node1 = clique[i];
                    for (unsigned int j = i + 1; j < size_clique; ++j) {
                        auto node2 = clique[j];
                        if (node1 > node2) swap(node1, node2); // Ensure node1 < node2
                        generated_edges.insert(encode_pair(node1, node2, n_nodes));
                    }

                }

            }
            for (const auto& pair_id : remaining_pairs) {
                generated_edges.insert(pair_id);
            }            
            for (unsigned long long pair_id : generated_edges) {
                auto edge = decode_pair(pair_id, n_nodes);
                outfile << edge.first << " " << edge.second << endl;
            }
            outfile.close();
        }
    }

    // Calculate and print statistics for generation times
    if (!generation_times.empty()) {
        double sum = 0.0;
        for (double time : generation_times) {
            sum += time;
        }
        double mean = sum / generation_times.size();

        double variance = 0.0;
        for (double time : generation_times) {
            variance += (time - mean) * (time - mean);
        }
        variance /= generation_times.size();
        double std_dev = sqrt(variance);

        cout << "\nGeneration time statistics:" << endl;
        cout << "Mean: " << fixed << setprecision(2) << mean << " ms" << endl;
        cout << "Std deviation: " << fixed << setprecision(2) << std_dev << " ms" << endl;
        cout << "Mean ± Std: " << fixed << setprecision(2) << mean << " ± " << std_dev << " ms" << endl;        
    }

    // Calculate and print statistics for write times
    if (!write_times.empty()) {
        double sum = 0.0;
        for (double time : write_times) {
            sum += time;
        }
        double mean = sum / write_times.size();
        double variance = 0.0;
        for (double time : write_times) {
            variance += (time - mean) * (time - mean);
        }
        variance /= write_times.size();
        double std_dev = sqrt(variance);

        cout << "\nWrite time statistics:" << endl;
        cout << "Mean: " << fixed << setprecision(2) << mean << " ms" << endl;
        cout << "Std deviation: " << fixed << setprecision(2) << std_dev << " ms" << endl;
        cout << "Mean ± Std: " << fixed << setprecision(2) << mean << " ± " << std_dev << " ms" << endl;
    }    

    // Final memory usage
    MemoryInfo final_memory = get_memory_usage();
    cout << endl;
    print_memory_info("Final memory usage", final_memory);
    
    return 0;
}