// underlying: Kronecker
// binding: iid
// input: (seed matrix, kronecker power, binding strength for node with different # zeros in their binary representation, # rounds)

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

void display_progress(int step, int total_steps, int bar_width = 50) {
    static auto start_time = chrono::steady_clock::now();
    auto current_time = chrono::steady_clock::now();
    chrono::duration<float> elapsed = current_time - start_time;
    float progress = static_cast<float>(step) / total_steps;
    int pos = static_cast<int>(bar_width * progress);

    // Calculate elapsed_time and remaining_time
    float elapsed_time = elapsed.count();
    float remaining_time = (elapsed_time / progress) - elapsed_time;

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
         << elapsed_time << " s, ETA: " << remaining_time << " s\r";
    cout.flush();
}

// input //
// n_nodes: number of nodes
// edge_probs: a 2D vector of edge probabilities
// alphas: a vector of binding strengths
// n_rounds: number of total rounds
// output //
// a list of edges
vector<int> generate_edges(int n_nodes, vector<vector<double>> edge_probs, vector<double> alphas, int n_rounds) {
    // initialize an empty edge list
    vector<int> edges;
    // we first compute the probability of each pair being sampled in each round
    // for each pair of nodes (u, v), if the original probability is p_uv=dict[u][v]
    // then the prob in each round is 1 - (1 - p_uv) ^ (1 / n_rounds)
    // we store the probability of each pair in a 2D vector
    vector<vector<double>> edge_probs_round(edge_probs.size(), vector<double>(edge_probs[0].size(), 0.0));
    // the remaining probability after all the rounds
    vector<vector<double>> edge_probs_remain(edge_probs.size(), vector<double>(edge_probs[0].size(), 0.0));
    vector<int> pairs_remain;
    // p_max = 1 - (1 - alpha^2) ^ n_rounds
    // auto p_max = 1.0 - pow(1.0 - alpha * alpha, n_rounds);
    // cout << "p_max: " << p_max << endl;
    vector<vector<int>> pairs_remain_omp(omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < edge_probs.size(); ++i) {
        for (int j = 0; j < edge_probs[i].size(); ++j) {
            int tid = omp_get_thread_num();
            auto r_ij = (1 - pow(1 - edge_probs[i][j], 1.0 / n_rounds)) / (alphas[i] * alphas[j]);
            if (r_ij > 1.0) {
                auto p_max = 1.0 - pow(1.0 - alphas[i] * alphas[j], n_rounds);
                edge_probs_round[i][j] = 1.0;
                edge_probs_remain[i][j] = 1.0 - (1.0 - edge_probs[i][j]) / (1.0 - p_max);
                // cout << "edge_probs[i][j]: " << edge_probs[i][j] << endl;
                // cout << "edge_probs_round[i][j]: " << edge_probs_round[i][j] << endl;
                // cout << "edge_probs_remain[i][j]: " << edge_probs_remain[i][j] << endl;
                if (i < j) {
                    int edge_ij = i * n_nodes + j;
                    pairs_remain_omp[tid].push_back(edge_ij);
                }
            } else {
                edge_probs_round[i][j] = r_ij;
            }
        }
    }

    // collect the remaining pairs
    for (const auto& p : pairs_remain_omp) {
        pairs_remain.insert(pairs_remain.end(), p.begin(), p.end());
    }

    // in total "n_rounds" rounds
    // in each round, we first sample nodes in "nodes", where each node is sampled with probabilitiy "alpha"
    // then we sample edges between the sampled nodes
    // we collect all the pairs between the sampled nodes, and find their probabilities in "prob_dict"
    // specifically, we first generate a random number "p_random" between 0 and 1
    // we add the pair (u, v) to the edge list if prob_dict[u][v] > p_random

    vector<vector<int>> edges_omp(omp_get_max_threads());
    std::atomic<int> progress(0);

#pragma omp parallel for
    for (int i = 0; i < n_rounds; ++i) {
        int tid = omp_get_thread_num();

        // random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(tid + seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        // sample nodes
        vector<int> sampled_nodes;
        for (int j = 0; j < n_nodes; ++j) {
            auto alpha_j = alphas[j];
            double random_v = dist(rng);
            if (random_v < alpha_j) {
                sampled_nodes.push_back(j);
            }
        }

        // sample edges
        double random_e = dist(rng);
        for (int j = 0; j < sampled_nodes.size(); ++j) {
            for (int k = j + 1; k < sampled_nodes.size(); ++k) {
                int edge_jk = sampled_nodes[j] * n_nodes + sampled_nodes[k];
                if (random_e < edge_probs_round[sampled_nodes[j]][sampled_nodes[k]]) {
                    edges_omp[tid].push_back(edge_jk);
                }
            }
        }

        progress++;
        if (tid == 0) {
            display_progress(progress, n_rounds);
        }
    }
    cout << endl;

    for (const auto& e : edges_omp) {
        edges.insert(edges.end(), e.begin(), e.end());
    }

    // deal with the remaining probability
    vector<vector<int>> edges_remain_omp(omp_get_max_threads());

#pragma omp parallel for
    for (int i = 0; i < pairs_remain.size(); ++i) {
        int tid = omp_get_thread_num();
        // random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        int edge_ij = pairs_remain[i];
        int i_node = edge_ij / n_nodes;
        int j_node = edge_ij % n_nodes;
        double random_p = dist(rng);
        if (random_p < edge_probs_remain[i_node][j_node]) {
            edges_remain_omp[tid].push_back(edge_ij);
        }
    }

    // collect the edges
    for (const auto& e : edges_remain_omp) {
        edges.insert(edges.end(), e.begin(), e.end());
    }

    return edges;
}

// given a interger and an order k, compute the number of zeros in its binary representation with length k
int count_zeros(int n, int k) {
    int count = 0;
    for (int i = 0; i < k; ++i) {
        if ((n & 1) == 0) {
            count++;
        }
        n >>= 1;
    }
    return count;
}

std::vector<std::vector<double>> kroneckerProduct(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    std::vector<std::vector<double>> result(a.size() * b.size(), std::vector<double>(a[0].size() * b[0].size()));

    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[0].size(); ++j) {
            for (std::size_t p = 0; p < b.size(); ++p) {
                for (std::size_t q = 0; q < b[0].size(); ++q) {
                    result[i * b.size() + p][j * b[0].size() + q] = a[i][j] * b[p][q];
                }
            }
        }
    }

    return result;
}

std::vector<std::vector<double>> kroneckerPower(const std::vector<std::vector<double>>& a, int power) {
    std::vector<std::vector<double>> result = a;

    for (int i = 1; i < power; ++i) {
        result = kroneckerProduct(result, a);
    }

    return result;
}

// main function
int main(int argc, char const* argv[]) {
    // read the arguments
    // argv[1]: input filename
    // argv[2]: output filename
    // argv[3]: number of generated graphs

    // read the file name from argv into a string
    if (argc != 4) {
        cout << "Usage: " << argv[0]
             << " <filename_in>"
             << " <filename_out>"
             << " <n_graphs>"
             << endl;
        return 1;
    }
    string filename_in = argv[1];
    string filename_out = argv[2];
    int n_graphs = atoi(argv[3]);

    // read the input file
    // first line: k, n_rounds, size of seed matrix
    // (k+1) lines after that: each line is the alpha for nodes with that number of zeros in their binary representation
    // the last lines: the seed matrix
    ifstream infile;
    infile.open(filename_in);
    int k_kron, n_rounds, size_seed;
    string line;
    getline(infile, line);
    istringstream iss(line);
    if (!(iss >> k_kron >> n_rounds >> size_seed)) {
        cout << "Error: the first line of the input file should contain k, n_rounds, size of seed matrix" << endl;
        return 1;
    }  // error
    cout << "k: " << k_kron << endl;
    cout << "n_rounds: " << n_rounds << endl;
    cout << "size_seed: " << size_seed << endl;

    double alpha;
    vector<double> alphas_zero;
    for (int i = 0; i < k_kron + 1; ++i) {
        getline(infile, line);
        istringstream iss(line);
        if (!(iss >> alpha)) {
            cout << "Error: the " << i + 2 << "th line of the input file should contain the alpha for nodes with that number of zeros in their binary representation" << endl;
            return 1;
        }  // error
        alphas_zero.push_back(alpha);
    }

    // print the alphas
    cout << "alphas: ";
    for (int i = 0; i < alphas_zero.size(); ++i) {
        cout << alphas_zero[i] << " ";
    }
    cout << endl;    

    // read the seed matrix
    vector<vector<double>> seed(size_seed, vector<double>(size_seed, 0));
    // each line has size_seed numbers, which is a row of the seed matrix
    for (int i = 0; i < size_seed; ++i) {
        getline(infile, line);
        istringstream iss(line);
        for (int j = 0; j < size_seed; ++j) {
            if (!(iss >> seed[i][j])) {
                cout << "Error: the " << i + 2 + k_kron << "th line of the input file should contain the " << j + 1 << "th row of the seed matrix" << endl;
                return 1;
            }  // error
        }
    }
    infile.close();

    int n_nodes = pow(size_seed, k_kron);

    // compute the kronecker power of seed matrix to obtain edge probabilities
    vector<vector<double>> edge_probs = kroneckerPower(seed, k_kron);
    int total_nodes = edge_probs.size();

    vector<double> alphas(n_nodes, 0.0);
    for(int i = 0; i < n_nodes; ++i) {
        alphas[i] = alphas_zero[count_zeros(i, k_kron)];
    }

    // generate n_graphs graphs
    for (size_t i_graph = 0; i_graph < n_graphs; i_graph++) {
        // generate edge using binding
        auto start = std::chrono::steady_clock::now();  // Start the timer
        auto edges = generate_edges(n_nodes, edge_probs, alphas, n_rounds);
        auto end = std::chrono::steady_clock::now();                                         // Stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  // Calculate the duration in milliseconds

        cout << endl;
        // print the number generated edges
        cout << "Number of generated edges with repetitions: " << edges.size() << endl;
        // print the time used for generating edges
        cout << "Time used for generating edges: " << duration.count() / 1000.0 << " seconds" << endl;

        auto start_time = std::chrono::steady_clock::now();

        // remove repeated edges
        sort(edges.begin(), edges.end());
        edges.erase(unique(edges.begin(), edges.end()), edges.end());

        // print the number generated edges
        cout << "Number of generated edges: " << edges.size() << endl;

        // collect the nodes involved in the edges
        vector<int> nodes_in_edges;
        for (int i = 0; i < edges.size(); ++i) {
            nodes_in_edges.push_back(edges[i] / total_nodes);
            nodes_in_edges.push_back(edges[i] % total_nodes);
        }
        // remove the repeated nodes
        sort(nodes_in_edges.begin(), nodes_in_edges.end());
        nodes_in_edges.erase(unique(nodes_in_edges.begin(), nodes_in_edges.end()), nodes_in_edges.end());
        // print the number of nodes
        cout << "Number of nodes: " << nodes_in_edges.size() << endl;

        auto end_time = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double time_used = duration.count() / 1000.0;
        cout << "Time used for removing duplications: " << time_used << " seconds" << endl;

        // write the edges in a file
        ofstream outfile;
        outfile.open(filename_out + "_" + to_string(i_graph) + ".txt");

        auto start_write = std::chrono::steady_clock::now();  // Start the timer

        for (int i = 0; i < edges.size(); ++i) {
            int node_i = edges[i] / total_nodes;
            int node_j = edges[i] % total_nodes;
            outfile << node_i << " " << node_j << endl;
        }

        auto end_write = std::chrono::steady_clock::now();                                                     // Stop the timer
        auto duration_write = std::chrono::duration_cast<std::chrono::milliseconds>(end_write - start_write);  // Calculate the duration in milliseconds
        double time_write = duration_write.count() / 1000.0;

        outfile.close();
    }
    return 0;
}
