#include <iostream>
#include <vector>
#include <bitset>
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>

const int MAX_N = 10;  // Maximum order size
using BitSet = std::bitset<MAX_N * (MAX_N-1) / 2>;
using Subset = std::vector<int>;

// Convert pair to bit position
inline int pair_to_bit(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    return i * n - (i * (i + 1)) / 2 + j - i - 1;
}

// Convert bit position back to pair (i, j) where i < j
inline std::pair<int, int> bit_to_pair(int bit, int n) {
    // Find i by solving: i * n - (i * (i + 1)) / 2 <= bit < (i+1) * n - ((i+1) * (i+2)) / 2
    int i = 0;
    while (i < n - 1) {
        int next_start = (i + 1) * n - ((i + 1) * (i + 2)) / 2;
        if (bit < next_start) break;
        i++;
    }
    // Now solve for j: bit = i * n - (i * (i + 1)) / 2 + j - i - 1
    int j = bit - (i * n - (i * (i + 1)) / 2) + i + 1;
    return std::make_pair(i, j);
}

// Generate all subsets of size >= 2
std::vector<Subset> generate_subsets(int n) {
    std::vector<Subset> subsets;
    for (int size = 2; size <= n; size++) {
        std::vector<bool> v(n);
        std::fill(v.end() - size, v.end(), true);
        do {
            Subset subset;
            for (int i = 0; i < n; i++) {
                if (v[i]) subset.push_back(i);
            }
            subsets.push_back(subset);
        } while (std::next_permutation(v.begin(), v.end()));
    }
    return subsets;
}

// Convert subset to bitset of pairs
BitSet subset_to_pairs(const Subset& subset, int n) {
    BitSet pairs;
    for (size_t i = 0; i < subset.size(); i++) {
        for (size_t j = i + 1; j < subset.size(); j++) {
            pairs.set(pair_to_bit(subset[i], subset[j], n));
        }
    }
    return pairs;
}

struct SearchResult {
    std::vector<Subset> sequence;
    std::vector<BitSet> coverage;
};

// Progress tracking
std::atomic<long long> total_sequences_found(0);
std::atomic<long long> total_nodes_explored(0);
std::atomic<long long> last_progress_update(0);
std::atomic<long long> last_progress_time(0);
std::chrono::time_point<std::chrono::steady_clock> start_time;
bool progress_enabled = false;
omp_lock_t progress_lock;

void print_progress() {
    if (!progress_enabled) return;
    
    // Only update progress if enough nodes have been explored since last update
    long long current_nodes = total_nodes_explored.load();
    long long last_update = last_progress_update.load();
    
    // Update every 1000 nodes or if it's been more than 2 seconds
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    long long last_time = last_progress_time.load();
    
    if (current_nodes - last_update < 1000 && elapsed - last_time < 2) {
        return;
    }
    
    // Try to update atomically
    long long expected = last_update;
    if (!last_progress_update.compare_exchange_weak(expected, current_nodes)) {
        return; // Another thread is updating
    }
    last_progress_time.store(elapsed);
    
    omp_set_lock(&progress_lock);
    
    long long sequences = total_sequences_found.load();
    
    // Clear the line and print progress
    std::cout << "\r" << std::string(80, ' ') << "\r";
    std::cout << "Progress: Sequences found: " << std::setw(10) << sequences 
              << " | Nodes explored: " << std::setw(10) << current_nodes
              << " | Time: " << std::setw(6) << elapsed << "s" << std::flush;
    
    omp_unset_lock(&progress_lock);
}

std::vector<SearchResult> recursive_search(
    const std::vector<Subset>& subsets,
    const std::vector<BitSet>& subset_pairs,
    const BitSet& pairs_remain,
    std::vector<Subset>& current_sequence,
    std::vector<BitSet>& current_coverage,
    std::vector<bool>& used,
    int n
) {
    std::vector<SearchResult> results;
    
    // Track nodes explored
    total_nodes_explored++;
    
    // Print progress periodically
    print_progress();
    
    // Accept current sequence as valid result if it's non-empty
    // (sequences don't need to cover all pairs anymore)
    if (!current_sequence.empty()) {
        SearchResult result;
        result.sequence = current_sequence;
        result.coverage = current_coverage;
        results.push_back(result);
        total_sequences_found++;
    }

    // Continue searching for longer sequences
    #pragma omp parallel for schedule(dynamic) if(current_sequence.empty())
    for (size_t i = 0; i < subsets.size(); i++) {
        if (used[i]) continue;

        // Each new subset must cover at least one new pair
        BitSet intersection = subset_pairs[i] & pairs_remain;
        if (intersection.none()) continue;

        std::vector<Subset> seq = current_sequence;
        std::vector<BitSet> cov = current_coverage;
        std::vector<bool> used_local = used;

        seq.push_back(subsets[i]);
        cov.push_back(intersection);
        used_local[i] = true;

        BitSet new_remain = pairs_remain;
        new_remain &= ~subset_pairs[i];

        auto sub_results = recursive_search(
            subsets, subset_pairs, new_remain,
            seq, cov, used_local, n
        );

        #pragma omp critical
        {
            results.insert(results.end(), sub_results.begin(), sub_results.end());
        }
    }

    return results;
}

std::vector<SearchResult> generate_cases(int n) {
    auto subsets = generate_subsets(n);
    std::vector<BitSet> subset_pairs;
    
    // Precompute all subset pairs
    for (const auto& subset : subsets) {
        subset_pairs.push_back(subset_to_pairs(subset, n));
    }

    // Initialize total pairs
    BitSet total_pairs;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            total_pairs.set(pair_to_bit(i, j, n));
        }
    }

    std::vector<Subset> current_sequence;
    std::vector<BitSet> current_coverage;
    std::vector<bool> used(subsets.size(), false);

    return recursive_search(
        subsets, subset_pairs, total_pairs,
        current_sequence, current_coverage, used, n
    );
}

int main(int argc, char* argv[]) {
    // int n_order = 4;
    int n_order;
    // reading argument from command line
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <n_order>" << std::endl;
        return 1;
    }
    n_order = atoi(argv[1]);

    
    omp_set_num_threads(omp_get_max_threads());
    
    // Initialize progress tracking
    omp_init_lock(&progress_lock);
    total_sequences_found = 0;
    total_nodes_explored = 0;
    last_progress_update = 0;
    last_progress_time = 0;
    start_time = std::chrono::steady_clock::now();
    progress_enabled = true;
    
    std::cout << "Starting search...\n";
    
    auto cases = generate_cases(n_order);
    
    // Clear progress line and print final summary
    std::cout << "\r" << std::string(80, ' ') << "\r";
    progress_enabled = false;
    omp_destroy_lock(&progress_lock);
    
    int cnt_dep = 0;
    int seq_index = 0;
    for (const auto& result : cases) {
        // Count total pairs covered by this sequence
        int total_pairs_covered = 0;
        for (const auto& cov : result.coverage) {
            total_pairs_covered += cov.count();
        }
        
        // Edge dependency exists when number of subsets < total pairs covered
        // (meaning some subsets cover multiple pairs)
        if (result.sequence.size() < total_pairs_covered) {
            cnt_dep++;
            seq_index++;
            
            std::cout << "Sequence " << seq_index << ": ";
            for (size_t idx = 0; idx < result.sequence.size(); idx++) {
                const auto& subset = result.sequence[idx];
                std::cout << "{ ";
                for (int x : subset) std::cout << x << " ";
                std::cout << "} ";
            }
            std::cout << "\n";
            
            // Print pair partition (only non-singletons; singletons are implicit)
            std::cout << "  Pair partition: ";
            bool has_non_singleton = false;
            for (size_t idx = 0; idx < result.coverage.size(); idx++) {
                const auto& cov = result.coverage[idx];
                // Only print if this subset covers more than one pair (non-singleton)
                if (cov.count() > 1) {
                    if (has_non_singleton) std::cout << " ";
                    has_non_singleton = true;
                    std::cout << "{ ";
                    bool first = true;
                    for (int bit = 0; bit < MAX_N * (MAX_N - 1) / 2; bit++) {
                        if (cov.test(bit)) {
                            if (!first) std::cout << ", ";
                            auto p = bit_to_pair(bit, n_order);
                            std::cout << "(" << p.first << "," << p.second << ")";
                            first = false;
                        }
                    }
                    std::cout << " }";
                }
            }
            if (!has_non_singleton) {
                std::cout << "(all singletons)";
            }
            std::cout << "\n";
        }
    }
    
    std::cout << "Found " << cases.size() << " valid sequences\n";
    std::cout << "Among them, " << cnt_dep << " cases involve edge dependency\n";
    
    return 0;
}