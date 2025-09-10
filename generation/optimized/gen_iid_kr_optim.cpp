// underlying: Stochastic Kronecker Model
// binding: parallel binding

#include <algorithm>
#include <atomic>
#include <boost/math/special_functions/binomial.hpp>
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
#include <unordered_map>
#include <vector>

using namespace std;

#define FREQ_WRITE 1000

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

// Helper function to encode ordered tuple (v1, v2, v3) into unsigned long long
inline unsigned long long encode_tuple(unsigned int v1, unsigned int v2, unsigned int v3,
                                       unsigned int n_nodes) {
    return static_cast<unsigned long long>(v1) * n_nodes * n_nodes +
           static_cast<unsigned long long>(v2) * n_nodes + v3;
}

// Helper function to decode tuple ID back to (v1, v2, v3)
inline tuple<unsigned int, unsigned int, unsigned int> decode_tuple(unsigned long long tuple_id,
                                                                    unsigned int n_nodes) {
    unsigned int v1 = static_cast<unsigned int>(tuple_id / (n_nodes * n_nodes));
    unsigned int remainder = static_cast<unsigned int>(tuple_id % (n_nodes * n_nodes));
    unsigned int v2 = remainder / n_nodes;
    unsigned int v3 = remainder % n_nodes;
    return make_tuple(v1, v2, v3);
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

// given a interger and an order k, compute the number of zeros in its binary representation with
// length k
unsigned int count_zeros(unsigned int n, unsigned int k) {
    unsigned int count = 0;
    for (unsigned int i = 0; i < k; ++i) {
        if ((n & 1) == 0) {
            count++;
        }
        n >>= 1;
    }
    return count;
}

// given two integers and an order k, compute the number of entries where (1) both are 0, (2) one is
// 0 and the other is 1, (3) both are 1
tuple<unsigned int, unsigned int, unsigned int> count_pattern(unsigned int n1, unsigned int n2,
                                                              unsigned int k) {
    unsigned int count_00 = 0;
    unsigned int count_01 = 0;
    unsigned int count_11 = 0;

    for (unsigned int i = 0; i < k; ++i) {
        if ((n1 & 1) == 0 && (n2 & 1) == 0) {
            count_00++;
        } else if ((n1 & 1) != (n2 & 1)) {
            count_01++;
        } else {
            count_11++;
        }
        n1 >>= 1;
        n2 >>= 1;
    }
    return make_tuple(count_00, count_01, count_11);
}

class Binomial {
    std::vector<std::vector<unsigned long long>> cache;

  public:
    Binomial(unsigned int maxN, unsigned int maxK) {
        cache.assign(maxN + 1, std::vector<unsigned long long>(maxK + 1, -1));
    }

    unsigned long long compute(unsigned int n, unsigned int k) {
        if (k > n)
            return 0;
        if (k == 0 || k == n)
            return 1;

        unsigned long long &res = cache[n][k];
        if (res != -1)
            return res; // return cached value

        res = compute(n - 1, k - 1) + compute(n - 1, k); // recursive formula
        return res;
    }
};

// Given a kron pattern, compute the number of node pairs with this kron pattern
unsigned long long count_node_pairs(unsigned int kron_pattern_id, unsigned int k_kron,
                                    Binomial &binomial) {
    auto pattern = decode_tuple(kron_pattern_id, k_kron + 1);
    auto count_00 = get<0>(pattern);
    auto count_01 = get<1>(pattern);
    auto count_11 = get<2>(pattern);

    // (k_kron choose count_00) * (k_kron - count_00 choose count_01)
    unsigned long long n_pairs =
        binomial.compute(k_kron, count_00) * binomial.compute(k_kron - count_00, count_01);
    return n_pairs;
}

// sample a random node pair with a given kron pattern
unsigned long long sample_node_pair(unsigned int kron_pattern_id, unsigned int n_nodes,
                                    unsigned int k_kron) {
    vector<unsigned int> entries;
    entries.reserve(k_kron + 1);
    for (unsigned int i = 0; i <= k_kron; ++i) {
        entries.push_back(i);
    }
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    // shuffle the entries
    shuffle(entries.begin(), entries.end(), rng);

    auto pattern = decode_tuple(kron_pattern_id, k_kron + 1);
    auto count_00 = get<0>(pattern);
    auto count_01 = get<1>(pattern);
    auto count_11 = get<2>(pattern);

    unsigned int node1 = 0;
    unsigned int node2 = 0;
    unsigned int cur_idx = 0;
    for (unsigned int i = 0; i < count_00; ++i) {
        cur_idx++;
    }
    for (unsigned int i = 0; i < count_01; ++i) {
        auto idx = entries[cur_idx];
        // node1 += 0 * (1 << idx);
        node2 += (1 << idx);
        cur_idx++;
    }
    for (unsigned int i = 0; i < count_11; ++i) {
        auto idx = entries[cur_idx];
        node1 += (1 << idx);
        node2 += (1 << idx);
        cur_idx++;
    }
    return encode_pair(node1, node2, n_nodes);
}

void generate_edges(unsigned int n_rounds, unsigned int n_nodes, unsigned int k_kron,
                    vector<vector<unsigned int>> &group_nodes, vector<float50> &alpha_groups,
                    vector<unsigned long long> &kron_patterns,
                    unordered_map<unsigned long long, unsigned long long> &pattern2n_pairs,
                    vector<float50> &kron_probs_round,
                    unordered_map<unsigned long long, float50> &kron_probs_remain,
                    string &output_file_name) {
    auto n_groups = group_nodes.size();
    std::atomic<unsigned int> progress(0);
    auto start_time = chrono::steady_clock::now();
    static mutex progress_mutex;

    vector<vector<unsigned long long>> edges_omp(omp_get_max_threads());

    // Open a file for each thread
    vector<ofstream> outfiles(omp_get_max_threads());    
    for (unsigned int tid = 0; tid < omp_get_max_threads(); ++tid) {
        outfiles[tid].open(output_file_name + "_" + to_string(tid) + ".txt");
    }

    // First, compute the number of rounds with no active pairs
    unsigned int n_rounds_no_active_pairs = 0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    double prob_min = kron_probs_round[0].convert_to<double>();
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

        // Periodically write the edges to the file to avoid Out of Memory
        if (i_round % FREQ_WRITE == 0 && !edges_omp[tid].empty()) {
            for (unsigned int i = 0; i < edges_omp[tid].size(); ++i) {
                auto edge = decode_pair(edges_omp[tid][i], n_nodes);
                outfiles[tid] << edge.first << " " << edge.second << endl;
            }
            edges_omp[tid].clear();
        }

        // random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(seed);
        uniform_real_distribution<double> dist(prob_min,
                                               1.0); // We have excluded rounds with no active pairs

        // Find active kron patterns
        float50 random_e = dist(rng);
        vector<unsigned long long> active_kron_patterns;
        active_kron_patterns.reserve(kron_patterns.size());
        // Find the first index i such that kron_probs_round[i] > random_e using binary search
        // And put all the following kron patterns into active_kron_patterns
        auto it = upper_bound(kron_probs_round.begin(), kron_probs_round.end(), random_e);
        auto index = distance(kron_probs_round.begin(), it);
        for (unsigned int i_kron_pattern = index; i_kron_pattern < kron_patterns.size();
             ++i_kron_pattern) {
            active_kron_patterns.push_back(kron_patterns[i_kron_pattern]);
        }

        // Skip if no active kron patterns (should not happen, but just keep it in case)
        if (active_kron_patterns.empty()) {
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

        // Merge the sampled nodes into a single vector
        vector<unsigned int> sampled_nodes_all;
        for (unsigned int i_group = 0; i_group < n_groups; ++i_group) {
            sampled_nodes_all.insert(sampled_nodes_all.end(), sampled_nodes[i_group].begin(),
                                     sampled_nodes[i_group].end());
        }
        
        // For each active kron pattern
        for (unsigned int i_kron_pattern = 0; i_kron_pattern < active_kron_patterns.size(); ++i_kron_pattern) {
            auto kron_pattern_id = active_kron_patterns[i_kron_pattern];
            auto pattern = decode_tuple(kron_pattern_id, k_kron + 1);
            auto count_00 = get<0>(pattern);
            auto count_01 = get<1>(pattern);
            auto count_11 = get<2>(pattern);
            
            // Identify the possible group pairs to satisfy this pattern
            // group id is the number of zeros in the binary representation
            unsigned int group_1 = count_00 + count_01;
            unsigned int group_2 = count_00;
            auto nodes_1 = group_nodes[group_1];
            auto nodes_2 = group_nodes[group_2];

            if (group_1 == group_2) {
                for (unsigned int i = 0; i < nodes_1.size(); ++i) {
                    for (unsigned int j = i + 1; j < nodes_1.size(); ++j) {
                        unsigned int node1 = nodes_1[i];
                        unsigned int node2 = nodes_1[j];
                        auto pattern_ij = count_pattern(node1, node2, k_kron);
                        if (pattern_ij == pattern) {
                            edges_omp[tid].push_back(encode_pair(node1, node2, n_nodes));
                        }
                    }
                }
            } else {
                for (unsigned int i = 0; i < nodes_1.size(); ++i) {
                    for (unsigned int j = 0; j < nodes_2.size(); ++j) {
                        unsigned int node1 = nodes_1[i];
                        unsigned int node2 = nodes_2[j];
                        auto pattern_ij = count_pattern(node1, node2, k_kron);
                        if (pattern_ij == pattern) {
                            edges_omp[tid].push_back(encode_pair(node1, node2, n_nodes));
                        }
                    }
                }
            }
        }
    }

    // Ensure we display 100% completion
    display_progress(n_rounds_gen, n_rounds_gen, start_time);
    cout << endl;

    // Ensure all the edges are written to the file
    #pragma omp parallel for
    for (unsigned int tid = 0; tid < omp_get_max_threads(); ++tid) {
        for (unsigned int i = 0; i < edges_omp[tid].size(); ++i) {
            auto edge = decode_pair(edges_omp[tid][i], n_nodes);
            outfiles[tid] << edge.first << " " << edge.second << endl;
        }
        edges_omp[tid].clear();
    }

    // Deal with reaminging edges
    if (!kron_probs_remain.empty()) {    
        for (auto it = kron_probs_remain.begin(); it != kron_probs_remain.end(); ++it) {
            auto kron_pattern_id = it->first;
            auto p_remain = it->second;
            auto n_pairs = pattern2n_pairs[kron_pattern_id];
            
            // Now we need to sample pairs, so that each pair is sampled with probability p_remain
            // To speed up this, we do independent sampling
            // In each round, we sample a pair uniformly at random
            // We now compute the number of rounds we need, so that each pair is sampled with probability p_remain
            // n_rounds_remain = log_{1 - 1/n_total_pairs} (1 - p_remain)

            float50 p_base = - 1.0 / n_pairs;
            float50 n_rounds_remain = log1p(-p_remain) / log1p(p_base);
            unsigned int n_rounds_remain_int = static_cast<unsigned int>(n_rounds_remain);

            #pragma omp parallel for schedule(dynamic, 32)
            for (int i_round = 0; i_round < n_rounds_remain_int; i_round++) {
                int tid = omp_get_thread_num();

                unsigned seed_thread = seed + i_round + tid * n_rounds_remain_int;
                mt19937 rng_thread(seed_thread);

                // sample a pair uniformly at random
                unsigned long long pair_id = sample_node_pair(kron_pattern_id, n_nodes, k_kron);
                edges_omp[tid].push_back(pair_id);
            }
            
            // Write the edges to the file
            #pragma omp parallel for
            for (unsigned int tid = 0; tid < omp_get_max_threads(); ++tid) {
                for (unsigned int i = 0; i < edges_omp[tid].size(); ++i) {
                    auto edge = decode_pair(edges_omp[tid][i], n_nodes);
                    outfiles[tid] << edge.first << " " << edge.second << endl;
                }
                edges_omp[tid].clear();
            }
        }
    }

    // Close all files
    for (unsigned int tid = 0; tid < omp_get_max_threads(); ++tid) {
        outfiles[tid].close();
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

    // read the file name from argv into a string
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " <filename_in>"
             << " <filename_out>"
             << " <n_graphs>"
             << " <up_scale>" << endl;
        return 1;
    }
    string filename_in = argv[1];
    string filename_out = argv[2];
    unsigned int n_graphs = atoi(argv[3]);
    unsigned int up_scale = atoi(argv[4]);

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

    unsigned int k_kron, n_rounds, size_seed;
    string line;
    getline(infile, line);
    istringstream iss(line);
    if (!(iss >> k_kron >> n_rounds >> size_seed)) {
        cout << "Error: the first line of the input file should contain k, n_rounds, size of seed "
                "matrix"
             << endl;
        return 1;
    } // error
    cout << "k: " << k_kron << endl;
    cout << "n_rounds: " << n_rounds << endl;
    cout << "size_seed: " << size_seed << endl;

    // after that, the (k+1) lines are the binding strength of each block
    float50 alpha;
    vector<float50> nzero2alpha;
    for (int i = 0; i < k_kron + 1; ++i) {
        getline(infile, line);
        istringstream iss(line);
        if (!(iss >> alpha)) {
            cout << "Error: the " << i + 2
                 << "th line of the input file should contain the alpha for nodes with that number "
                    "of zeros in their binary representation"
                 << endl;
            return 1;
        } // error
        nzero2alpha.push_back(alpha);
    }

    // read the seed matrix
    vector<vector<float50>> seed(size_seed, vector<float50>(size_seed, 0));
    // each line has size_seed numbers, which is a row of the seed matrix
    for (int i = 0; i < size_seed; ++i) {
        getline(infile, line);
        istringstream iss(line);
        for (int j = 0; j < size_seed; ++j) {
            if (!(iss >> seed[i][j])) {
                cout << "Error: the " << i + 2 + k_kron
                     << "th line of the input file should contain the " << j + 1
                     << "th row of the seed matrix" << endl;
                return 1;
            } // error
        }
    }
    infile.close();

    k_kron += up_scale;

    Binomial binomial(k_kron, k_kron);

    // for the upsacled part, fill in the alpha values as the average of the alpha values
    float50 alpha_avg = 0.0;
    for (int i = 0; i < nzero2alpha.size(); ++i) {
        alpha_avg += nzero2alpha[i];
    }
    alpha_avg /= nzero2alpha.size();
    for (int i = 0; i < up_scale; ++i) {
        nzero2alpha.push_back(alpha_avg);
    }

    unsigned int n_nodes = pow(size_seed, k_kron);

    cout << "Number of rounds: " << n_rounds << endl;
    cout << "Number of nodes: " << n_nodes << endl;

    unsigned int n_groups = nzero2alpha.size();
    vector<vector<unsigned int>> nzero2nodes(n_groups);
    for (unsigned int i = 0; i < n_nodes; ++i) {
        unsigned int n_zeros_i = count_zeros(i, k_kron);
        nzero2nodes[n_zeros_i].push_back(i);
    }

    unsigned int total_nodes = n_nodes;

    // Prepare an unordered_map: From (#00, #01, #11) to the corresponding edge probability
    // Which is (seed[0][0] ** #00) * (seed[0][1] ** #01) * (seed[1][1] ** #11)
    vector<unsigned long long> kron_patterns;
    // vector<float50> kron_probs;
    vector<float50> kron_probs_round;
    unordered_map<unsigned long long, unsigned long long> pattern2n_pairs;
    unordered_map<unsigned long long, float50> kron_probs_remain;

    // Enumerate all tuples (i, j, k) with i + j + k = k_kron
    for (unsigned int i = 0; i <= k_kron; ++i) {
        for (unsigned int j = 0; j <= k_kron - i; ++j) {
            unsigned int k = k_kron - i - j;
            unsigned long long kron_pattern_id = encode_tuple(i, j, k, k_kron + 1);
            kron_patterns.push_back(kron_pattern_id);
            pattern2n_pairs[kron_pattern_id] = count_node_pairs(kron_pattern_id, k_kron, binomial);
            float50 prob = 1.0;
            prob *= pow(seed[0][0], i);
            prob *= pow(seed[0][1], j);
            prob *= pow(seed[1][1], k);
            // kron_probs.push_back(prob);
            // Compute the probability for each round
            float50 r_ij = (1 - pow(1 - prob, 1.0 / n_rounds)) / (nzero2alpha[i] * nzero2alpha[j]);
            if (r_ij > 1.0) {
                float50 p_max = 1.0 - pow(1.0 - nzero2alpha[i] * nzero2alpha[j], n_rounds);
                kron_probs_round.push_back(1.0);
                kron_probs_remain[kron_patterns[i]] = 1.0 - (1.0 - prob) / (1.0 - p_max);
            } else {
                kron_probs_round.push_back(r_ij);
            }
        }
    }
    // Sort kron_patterns and kron_probs based on kron_probs
    vector<size_t> indices(kron_patterns.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    sort(indices.begin(), indices.end(),
         [&](size_t a, size_t b) { return kron_probs_round[a] < kron_probs_round[b]; });
    vector<unsigned long long> kron_patterns_sorted(kron_patterns.size());
    vector<float50> kron_probs_sorted(kron_probs_round.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        kron_patterns_sorted[i] = kron_patterns[indices[i]];
        kron_probs_sorted[i] = kron_probs_round[indices[i]];
    }
    kron_patterns = std::move(kron_patterns_sorted);
    kron_probs_round = std::move(kron_probs_sorted);

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
        
        string file_name = filename_out + "_" + to_string(i_graph);

        generate_edges(n_rounds, total_nodes, k_kron, nzero2nodes, nzero2alpha, kron_patterns,
                       pattern2n_pairs, kron_probs_round, kron_probs_remain, file_name);        

        auto end = std::chrono::steady_clock::now(); // Stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start); // Calculate the duration in milliseconds
        double generation_time_ms = duration.count();
        generation_times.push_back(generation_time_ms);

        // Track memory after generation
        MemoryInfo post_generation_memory = get_memory_usage();
        memory_snapshots.push_back(post_generation_memory);
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
