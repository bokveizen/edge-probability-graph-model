// underlying: Stochastic Block Model
// binding: parallel binding

#include <algorithm>
#include <atomic>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

typedef boost::multiprecision::cpp_dec_float_50 float50; // 50 digits of precision

void display_progress(int step, int total_steps, chrono::steady_clock::time_point start_time,
                      int bar_width = 50) {
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
    cout << "] " << fixed << setprecision(2) << progress * 100.0 << " %, Elapsed: " << step << "/"
         << total_steps << ", " << elapsed_time << " s";

    if (remaining_time > 0) {
        cout << ", ETA: " << remaining_time << " s";
    }

    cout << "\r";
    cout.flush();
}

// Helper function to encode pair (v1, v2) where v1 < v2 into unsigned long long
inline unsigned long long encode_pair(unsigned int v1, unsigned int v2, unsigned int n_nodes) {
    if (v1 > v2)
        swap(v1, v2);
    return static_cast<unsigned long long>(v1) * n_nodes + v2;
}

// Helper function to decode pair ID back to (v1, v2)
inline pair<unsigned int, unsigned int> decode_pair(unsigned long long pair_id,
                                                    unsigned int n_nodes) {
    unsigned int v1 = static_cast<unsigned int>(pair_id / n_nodes);
    unsigned int v2 = static_cast<unsigned int>(pair_id % n_nodes);
    return make_pair(v1, v2);
}

// Memory tracking functions
struct MemoryInfo {
    size_t vm_peak_kb = 0; // Peak virtual memory
    size_t vm_size_kb = 0; // Current virtual memory
    size_t vm_rss_kb = 0;  // Current physical memory (RSS)
    size_t vm_hwm_kb = 0;  // Peak physical memory
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

void print_memory_info(const string &label, const MemoryInfo &info) {
    cout << label << ":" << endl;
    cout << "  Current RSS: " << fixed << setprecision(2) << info.vm_rss_kb / 1024.0 << " MB"
         << endl;
    cout << "  Peak RSS: " << fixed << setprecision(2) << info.vm_hwm_kb / 1024.0 << " MB" << endl;
    cout << "  Current Virtual: " << fixed << setprecision(2) << info.vm_size_kb / 1024.0 << " MB"
         << endl;
    cout << "  Peak Virtual: " << fixed << setprecision(2) << info.vm_peak_kb / 1024.0 << " MB"
         << endl;
}

void generate_edges(unsigned int n_rounds, unsigned int n_nodes, vector<vector<unsigned int>> &group_nodes,
                    vector<float50> &alpha_groups, vector<unsigned long long> &pairs_sorted,
                    vector<float50> &probs_sorted, vector<vector<float50>> &edge_probs_remain,
                    vector<unsigned long long> &pairs_remain,
                    vector<pair<vector<unsigned int>, vector<unsigned int>>> &bicliques,
                    vector<vector<unsigned int>> &cliques,
                    vector<unsigned long long> &remaining_edges) {
    auto n_groups = group_nodes.size();
    std::atomic<unsigned int> progress(0);
    auto start_time = chrono::steady_clock::now();
    static mutex progress_mutex;

    vector<vector<pair<vector<unsigned int>, vector<unsigned int>>>> bicliques_omp(
        omp_get_max_threads());
    vector<vector<vector<unsigned int>>> cliques_omp(omp_get_max_threads());
    vector<vector<unsigned long long>> remaining_edges_omp(omp_get_max_threads());

    // First, compute the number of rounds with no active pairs
    unsigned int n_rounds_no_active_pairs = 0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    double prob_min = probs_sorted[0].convert_to<double>();
    binomial_distribution<unsigned int> dist_binom(n_rounds, prob_min);
    n_rounds_no_active_pairs = dist_binom(rng);
    unsigned int n_rounds_gen = n_rounds - n_rounds_no_active_pairs;

    #pragma omp parallel for schedule(dynamic, 32)
    for (unsigned int i_round = 0; i_round < n_rounds_gen; ++i_round) {
        int current_progress = progress;
        unsigned int tid = omp_get_thread_num();
        
        if (current_progress % max(1U, n_rounds_gen / 100U) == 0 || tid == 0) {
            lock_guard<mutex> lock(progress_mutex);
            display_progress(current_progress, n_rounds_gen, start_time);
        }
        progress++;

        // random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(seed);
        uniform_real_distribution<double> dist(prob_min,
                                               1.0); // We have excluded rounds with no active pairs

        // Find active group pairs
        float50 random_e = dist(rng);
        vector<unsigned long long> active_pairs;
        active_pairs.reserve(pairs_sorted.size());
        // Find the first index i such that probs_sorted[i] > random_e using binary search
        // And put all the following pairs into active_pairs
        auto it = upper_bound(probs_sorted.begin(), probs_sorted.end(), random_e);
        auto index = distance(probs_sorted.begin(), it);
        for (unsigned int i_pair = index; i_pair < pairs_sorted.size(); ++i_pair) {
            active_pairs.push_back(pairs_sorted[i_pair]);
        }

        // Skip if no active pairs (should not happen, but just keep it in case)
        if (active_pairs.empty()) {
            continue;
        }

        vector<vector<unsigned int>> sampled_nodes(n_groups);
        // Sample nodes from each group
        for (unsigned int i_group = 0; i_group < n_groups; ++i_group) {
            auto nodes_i = group_nodes[i_group];
            auto n_nodes_i = nodes_i.size();
            auto alpha_i = alpha_groups[i_group];
            // sample from nodes_i independently each with probability alpha_i
            // First, determine how many nodes to sample using binomial distribution
            double alpha_double = alpha_i.convert_to<double>();
            binomial_distribution<unsigned int> dist_binom_nodes(n_nodes_i, alpha_double);
            unsigned int n_nodes_to_sample = dist_binom_nodes(rng);

            unordered_set<unsigned int> selected_nodes_i;
            selected_nodes_i.reserve(n_nodes_to_sample);

            if (n_nodes_to_sample <= n_nodes_i / 2) {
                // When sampling few nodes, use rejection sampling
                uniform_int_distribution<unsigned int> node_dist(0, n_nodes_i - 1);
                while (selected_nodes_i.size() < n_nodes_to_sample) {
                    unsigned int node = node_dist(rng);
                    selected_nodes_i.insert(node);
                }
            } else {
                // When sampling many nodes, sample the complement (nodes to exclude)
                int n_nodes_to_exclude = n_nodes_i - n_nodes_to_sample;
                unordered_set<unsigned int> excluded_nodes;
                excluded_nodes.reserve(n_nodes_to_exclude);
                uniform_int_distribution<unsigned int> node_dist(0, n_nodes_i - 1);
                while (excluded_nodes.size() < n_nodes_to_exclude) {
                    unsigned int node = node_dist(rng);
                    excluded_nodes.insert(node);
                }
                // Add all non-excluded nodes
                for (unsigned int j = 0; j < n_nodes_i; ++j) {
                    if (excluded_nodes.find(j) == excluded_nodes.end()) {
                        selected_nodes_i.insert(j);
                    }
                }
            }
            sampled_nodes[i_group] =
                vector<unsigned int>(selected_nodes_i.begin(), selected_nodes_i.end());
        }

        // Add edges for each active group pair
        for (unsigned int i_pair = 0; i_pair < active_pairs.size(); ++i_pair) {
            auto pair_id = active_pairs[i_pair];
            auto group_pair = decode_pair(pair_id, n_groups);
            auto group_1 = group_pair.first;
            auto group_2 = group_pair.second;
            if (group_1 == group_2) {
                if (sampled_nodes[group_1].size() > 1) {
                    cliques_omp[tid].push_back(sampled_nodes[group_1]);
                }
            } else {
                if (sampled_nodes[group_1].size() > 0 && sampled_nodes[group_2].size() > 0) {
                    bicliques_omp[tid].push_back(
                        make_pair(sampled_nodes[group_1], sampled_nodes[group_2]));
                }
            }
        }
    }

    // Ensure we display 100% completion
    display_progress(n_rounds_gen, n_rounds_gen, start_time);
    cout << endl;

    // Collect results from all threads
    for (const auto& thread_cliques : cliques_omp) {
        cliques.insert(cliques.end(), thread_cliques.begin(), thread_cliques.end());
    }
    for (const auto& thread_bicliques : bicliques_omp) {
        bicliques.insert(bicliques.end(), thread_bicliques.begin(), thread_bicliques.end());
    }

    // Deal with remaining edges
    if (!pairs_remain.empty()) {        
                
        for (unsigned int i = 0; i < pairs_remain.size(); ++i) {
            unsigned int tid = omp_get_thread_num();
            auto pair_id = pairs_remain[i];
            auto group_pair = decode_pair(pair_id, n_groups);
            auto group_1 = group_pair.first;
            auto group_2 = group_pair.second;
            if (group_1 == group_2) {  // Sample edges with in a group
                auto n_nodes_1 = group_nodes[group_1].size();
                unsigned long long n_total_pairs = n_nodes_1 * (n_nodes_1 - 1) / 2;
                auto p_remain = edge_probs_remain[group_1][group_1];

                // Now we need to sample pairs, so that each pair is sampled with probability p_remain
                // To speed up this, we do independent sampling
                // In each round, we sample a pair uniformly at random
                // We now compute the number of rounds we need, so that each pair is sampled with probability p_remain
                // n_rounds_remain = log_{1 - 1/n_total_pairs} (1 - p_remain)

                float50 p_base = - 1.0 / n_total_pairs;
                float50 n_rounds_remain = log1p(-p_remain) / log1p(p_base);
                
                unsigned int n_rounds_remain_int = static_cast<unsigned int>(n_rounds_remain);                

                #pragma omp parallel for schedule(dynamic, 32)
                for (int i_round = 0; i_round < n_rounds_remain_int; i_round++) {
                    int tid = omp_get_thread_num();
                    
                    // Each thread needs its own RNG with a unique seed
                    unsigned seed_thread = seed + i_round + tid * n_rounds_remain_int;
                    mt19937 rng_thread(seed_thread);

                    // sample a pair uniformly at random
                    unsigned int node1, node2;
                    do {
                        uniform_int_distribution<unsigned int> node_dist(0, n_nodes_1 - 1);
                        node1 = node_dist(rng_thread);
                        node2 = node_dist(rng_thread);
                    } while (node1 == node2);
                    node1 = group_nodes[group_1][node1];
                    node2 = group_nodes[group_1][node2];
                    if (node1 > node2) swap(node1, node2); // Ensure node1 < node2
                    unsigned long long pair_id = encode_pair(node1, node2, n_nodes);
                    remaining_edges_omp[tid].push_back(pair_id);
                }
            } else {  // Sample edges between two groups
                auto n_nodes_1 = group_nodes[group_1].size();
                auto n_nodes_2 = group_nodes[group_2].size();
                unsigned long long n_total_pairs = n_nodes_1 * n_nodes_2;
                auto p_remain = edge_probs_remain[group_1][group_2];

                // Now we need to sample pairs, so that each pair is sampled with probability p_remain
                // To speed up this, we do independent sampling
                // In each round, we sample a pair uniformly at random
                // We now compute the number of rounds we need, so that each pair is sampled with probability p_remain
                // n_rounds_remain = log_{1 - 1/n_total_pairs} (1 - p_remain)

                float50 p_base = - 1.0 / n_total_pairs;
                float50 n_rounds_remain = log1p(-p_remain) / log1p(p_base);
                
                unsigned int n_rounds_remain_int = static_cast<unsigned int>(n_rounds_remain);
                
                #pragma omp parallel for schedule(dynamic, 32)
                for (int i_round = 0; i_round < n_rounds_remain_int; i_round++) {
                    int tid = omp_get_thread_num();

                    // Each thread needs its own RNG with a unique seed
                    unsigned seed_thread = seed + i_round + tid * n_rounds_remain_int;
                    mt19937 rng_thread(seed_thread);

                    // sample a pair uniformly at random
                    unsigned int node1, node2;
                    uniform_int_distribution<unsigned int> node_dist_1(0, n_nodes_1 - 1);
                    uniform_int_distribution<unsigned int> node_dist_2(0, n_nodes_2 - 1);
                    node1 = node_dist_1(rng_thread);
                    node2 = node_dist_2(rng_thread);
                    node1 = group_nodes[group_1][node1];
                    node2 = group_nodes[group_2][node2];
                    if (node1 > node2) swap(node1, node2); // Ensure node1 < node2
                    unsigned long long pair_id = encode_pair(node1, node2, n_nodes);
                    remaining_edges_omp[tid].push_back(pair_id);
                }                
            }
        }

        // Collect results from all threads
        for (const auto& thread_edges : remaining_edges_omp) {
            remaining_edges.insert(remaining_edges.end(), thread_edges.begin(), thread_edges.end());
        }
    }
    return;
}

// main function
int main(int argc, char const *argv[]) {
    // read the arguments
    // argv[1]: input filename
    // argv[2]: output filename
    // argv[3]: number of generated graphs
    // argv[4]: up_scale
    // argv[5]: expand

    // read the file name from argv into a string
    if (argc != 6) {
        cout << "Usage: " << argv[0] << " <filename_in>"
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

    // read the input file: number of nodes, density, binding strength, number of
    // rounds, probability in each round
    ifstream infile;
    infile.open(filename_in);
    if (!infile.is_open()) {
        cerr << "Error: Cannot open input file " << filename_in << endl;
        return 1;
    }

    unsigned int n_blocks, n_rounds;
    string line;
    getline(infile, line);
    istringstream iss(line);
    if (!(iss >> n_blocks >> n_rounds)) {
        cout << "Error: cannot read the numbers of blocks and rounds" << endl;
        return 1;
    }
    // after that, the n_blocks lines are the binding strength of each block
    vector<float50> block2alpha;
    for (unsigned int i = 0; i < n_blocks; ++i) {
        getline(infile, line);
        istringstream iss(line);
        float50 alpha;
        if (!(iss >> alpha)) {
            cout << "Error: cannot read the binding strength of block " << i << endl;
            return 1;
        }
        block2alpha.push_back(alpha);
    }
    // after that, the n_blocks lines are the probabilities of each block
    // each line contains n_blocks numbers
    vector<vector<float50>> p_blocks(n_blocks, vector<float50>(n_blocks, 0.0));
    for (unsigned int i = 0; i < n_blocks; ++i) {
        getline(infile, line);
        istringstream iss(line);
        for (unsigned int j = 0; j < n_blocks; ++j) {
            if (!(iss >> p_blocks[i][j])) {
                cout << "Error: cannot read the probability of block " << i << " and " << j << endl;
                return 1;
            }
        }
    }
    // finally, the n_blocks lines are the number of nodes in each block
    vector<vector<unsigned int>> block2nodes(n_blocks);
    unsigned int cur_node = 0;
    for (unsigned int i = 0; i < n_blocks; ++i) {
        getline(infile, line);
        istringstream iss(line);
        unsigned int n_nodes;
        if (!(iss >> n_nodes)) {
            cout << "Error: cannot read the number of nodes in block " << i << endl;
            return 1;
        }
        n_nodes = n_nodes * up_scale;
        for (unsigned int j = 0; j < n_nodes; ++j) {
            block2nodes[i].push_back(cur_node);
            cur_node++;
        }
    }
    unsigned int total_nodes = cur_node;
    unsigned int n_nodes = total_nodes;
    infile.close();
    cout << "Number of blocks: " << n_blocks << endl;
    cout << "Number of rounds: " << n_rounds << endl;
    cout << "Total number of nodes: " << total_nodes << endl;    

    vector<vector<float50>> edge_probs_round(n_blocks, vector<float50>(n_blocks, 0.0));
    vector<vector<float50>> edge_probs_remain(n_blocks, vector<float50>(n_blocks, 0.0));
    for (unsigned int i = 0; i < n_blocks; ++i) {
        for (unsigned int j = i; j < n_blocks; ++j) {
            float50 edge_prob = p_blocks[i][j];
            if (edge_prob > 1.0) {
                edge_prob = 1.0;
            }
            // Compute the probability for each round
            float50 r_ij =
                (1 - pow(1 - edge_prob, 1.0 / n_rounds)) / (block2alpha[i] * block2alpha[j]);
            if (r_ij > 1.0) {
                float50 p_max = 1.0 - pow(1.0 - block2alpha[i] * block2alpha[j], n_rounds);
                edge_probs_round[i][j] = 1.0;
                edge_probs_remain[i][j] = 1.0 - (1.0 - edge_prob) / (1.0 - p_max);
            } else {
                edge_probs_round[i][j] = r_ij;
            }
        }
    }

    // Now, based on edge_probs_round, we sort the (i, j) pairs w.r.t. edge_probs_round
    // This gives us two vectors:
    // pairs_sorted and probs_sorted
    // Also, collect the (i, j) pairs with non-zero edge_probs_remain[i][j]    

    vector<unsigned long long> pairs_sorted;
    vector<float50> probs_sorted;
    pairs_sorted.reserve(n_blocks * (n_blocks + 1) / 2);
    probs_sorted.reserve(n_blocks * (n_blocks + 1) / 2);
    vector<unsigned long long> pairs_remain;
    pairs_remain.reserve(n_blocks * (n_blocks + 1) / 2);
    for (unsigned int i = 0; i < n_blocks; ++i) {
        for (unsigned int j = i; j < n_blocks; ++j) {
            pairs_sorted.push_back(encode_pair(i, j, n_blocks));
            probs_sorted.push_back(edge_probs_round[i][j]);
            if (edge_probs_remain[i][j] > 0.0) {
                pairs_remain.push_back(encode_pair(i, j, n_blocks));
            }
        }
    }    
    
    // Create indices vector to sort pairs and probabilities together
    vector<size_t> indices(pairs_sorted.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    
    // Sort indices based on probabilities (ascending order)
    sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return probs_sorted[a] < probs_sorted[b];
    });
    
    // Reorder both vectors based on sorted indices
    vector<unsigned long long> pairs_sorted_temp(pairs_sorted.size());
    vector<float50> probs_sorted_temp(probs_sorted.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        pairs_sorted_temp[i] = pairs_sorted[indices[i]];
        probs_sorted_temp[i] = probs_sorted[indices[i]];
    }
    
    pairs_sorted = std::move(pairs_sorted_temp);
    probs_sorted = std::move(probs_sorted_temp);

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
            cout << "Progress: " << i_graph << "/" << n_graphs << " graphs, Current RSS: " << fixed
                 << setprecision(1) << current_memory.vm_rss_kb / 1024.0 << " MB" << endl;
        }

        // generate edge using binding
        auto start = std::chrono::steady_clock::now(); // Start the timer

        vector<pair<vector<unsigned int>, vector<unsigned int>>> bicliques;
        vector<vector<unsigned int>> cliques;
        vector<unsigned long long> remaining_edges;

        generate_edges(n_rounds, total_nodes, block2nodes, block2alpha, pairs_sorted, probs_sorted,
                       edge_probs_remain, pairs_remain, bicliques, cliques, remaining_edges);

        auto end = std::chrono::steady_clock::now(); // Stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start); // Calculate the duration in milliseconds
        double generation_time_ms = duration.count();
        generation_times.push_back(generation_time_ms);

        // Track memory after generation
        MemoryInfo post_generation_memory = get_memory_usage();
        memory_snapshots.push_back(post_generation_memory);

        // Write the cliques, bicliques, and remaining edges directly (much more efficient!)
        ofstream outfile;
        outfile.open(filename_out + "_" + to_string(i_graph) + ".txt");
        auto start_write = std::chrono::steady_clock::now(); // Start the timer

        // Write cliques section
        outfile << "CLIQUES " << cliques.size() << endl;
        for (const auto &clique : cliques) {
            outfile << clique.size();
            for (unsigned int node : clique) {
                outfile << " " << node;
            }
            outfile << endl;
        }

        // Write bicliques section
        outfile << "BICLIQUES " << bicliques.size() << endl;
        for (const auto &biclique : bicliques) {
            outfile << biclique.first.size() << " " << biclique.second.size();
            for (unsigned int node : biclique.first) {
                outfile << " " << node;
            }
            for (unsigned int node : biclique.second) {
                outfile << " " << node;
            }
            outfile << endl;
        }

        // Write remaining edges section
        outfile << "REMAINING_EDGES " << remaining_edges.size() << endl;
        for (unsigned long long pair_id : remaining_edges) {
            auto edge = decode_pair(pair_id, n_nodes);
            outfile << edge.first << " " << edge.second << endl;
        }

        auto end_write = std::chrono::steady_clock::now(); // Stop the timer
        auto duration_write = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_write - start_write); // Calculate the duration in milliseconds
        double write_time_ms = duration_write.count();
        write_times.push_back(write_time_ms);

        outfile.close();

        if (expand) {
            ofstream outfile;
            outfile.open(filename_out + "_" + to_string(i_graph) + "_expand.txt");
            unordered_set<unsigned long long> generated_edges;
            // Expand cliques
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
            // Expand bi-cliques
            for (const auto& biclique : bicliques) {
                vector<unsigned int> nodes_1 = biclique.first;
                vector<unsigned int> nodes_2 = biclique.second;                
                auto size_1 = nodes_1.size();
                auto size_2 = nodes_2.size();
                for (unsigned int i = 0; i < size_1; ++i) {
                    for (unsigned int j = 0; j < size_2; ++j) {
                        auto node1 = nodes_1[i];
                        auto node2 = nodes_2[j];
                        if (node1 > node2) swap(node1, node2); // Ensure node1 < node2
                        generated_edges.insert(encode_pair(node1, node2, n_nodes));
                    }
                }
            }
            // Remaining edges
            for (unsigned long long pair_id : remaining_edges) {
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
        cout << "Mean ± Std: " << fixed << setprecision(2) << mean << " ± " << std_dev << " ms"
             << endl;
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
        cout << "Mean ± Std: " << fixed << setprecision(2) << mean << " ± " << std_dev << " ms"
             << endl;
    }

    // Final memory usage
    MemoryInfo final_memory = get_memory_usage();
    cout << endl;
    print_memory_info("Final memory usage", final_memory);

    return 0;
}
