// underlying: SBM
// binding: iid
// input: (# blocks, # rounds, binding strength per block, p_blocks, N_blocks)

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
    display_progress(progress, n_rounds);
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
    // first line: n_blocks, n_round    
    ifstream infile;
    infile.open(filename_in);
    int n_blocks, n_rounds;
    string line;
    getline(infile, line);
    istringstream iss(line);
    if (!(iss >> n_blocks >> n_rounds)) {
        cout << "Error: cannot read the numbers of blocks and rounds" << endl;
        return 1;
    }    
    // after that, the n_blocks lines are the binding strength of each block    
    vector<double> alphas_blocks;
    for (int i = 0; i < n_blocks; ++i) {
        getline(infile, line);
        istringstream iss(line);
        double alpha;
        if (!(iss >> alpha)) {
            cout << "Error: cannot read the binding strength of block " << i << endl;
            return 1;
        }
        alphas_blocks.push_back(alpha);
    }
    // after that, the n_blocks lines are the probabilities of each block
    // each line contains n_blocks numbers
    vector<vector<double>> p_blocks(n_blocks, vector<double>(n_blocks, 0.0));
    for (int i = 0; i < n_blocks; ++i) {
        getline(infile, line);
        istringstream iss(line);
        for (int j = 0; j < n_blocks; ++j) {
            if (!(iss >> p_blocks[i][j])) {
                cout << "Error: cannot read the probability of block " << i << " and " << j << endl;
                return 1;
            }
        }
    }
    // finally, the n_blocks lines are the number of nodes in each block
    vector<int> n_nodes_blocks;
    for (int i = 0; i < n_blocks; ++i) {
        getline(infile, line);
        istringstream iss(line);
        int n_nodes;
        if (!(iss >> n_nodes)) {
            cout << "Error: cannot read the number of nodes in block " << i << endl;
            return 1;
        }
        n_nodes_blocks.push_back(n_nodes);
    }
    infile.close();

    // calculate the total number of nodes and the offset of each block (i.e., the first node of each point)
    int total_nodes = 0;
    vector<int> offset_blocks(n_blocks, 0);
    for (int i = 0; i < n_blocks; ++i) {
        total_nodes += n_nodes_blocks[i];
        if (i > 0) {
            offset_blocks[i] = offset_blocks[i - 1] + n_nodes_blocks[i - 1];
        }
    }
    // generate the edge probability matrix based on p_blocks and n_blocks
    vector<vector<double>> edge_probs(total_nodes, vector<double>(total_nodes, 0.0));
    for (int i = 0; i < n_blocks; ++i) {
        for (int j = 0; j < n_blocks; ++j) {
            for (int k = 0; k < n_nodes_blocks[i]; ++k) {
                for (int l = 0; l < n_nodes_blocks[j]; ++l) {
                    edge_probs[k + offset_blocks[i]][l + offset_blocks[j]] = p_blocks[i][j];
                    // symmetric
                    edge_probs[l + offset_blocks[j]][k + offset_blocks[i]] = p_blocks[i][j];
                }
            }
        }
    }    
    // generate the alphas vector
    vector<double> alphas(total_nodes, 0.0);
    for (int i = 0; i < n_blocks; ++i) {
        for (int j = 0; j < n_nodes_blocks[i]; ++j) {
            alphas[j + offset_blocks[i]] = alphas_blocks[i];
        }
    }

    // generate n_graphs graphs
    for (size_t i_graph = 0; i_graph < n_graphs; i_graph++) {
        // generate edge using binding
        auto start = std::chrono::steady_clock::now();  // Start the timer
        auto edges = generate_edges(total_nodes, edge_probs, alphas, n_rounds);
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
