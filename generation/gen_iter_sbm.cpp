// underlying: SBM
// binding: iterative exhaustive
// input: (# blocks, # rounds, binding strength per block, p_blocks, N_blocks)

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

double epsilon = 1e-6;

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

// input //
// n_nodes: number of nodes
// edge_probs: a 2D vector of edge probabilities
// alphas: a vector of binding strengths
// n_rounds: number of total rounds
// output //
// a list of edges
// input //
// n_nodes: number of nodes
// output //
// a list of edges
vector<int> generate_edges(int n_nodes, vector<vector<double>> edge_probs, vector<double> alphas, int n_rounds) {
    // initialize an empty edge list
    vector<int> edges;
    int total_size = n_nodes * (n_nodes - 1) / 2;

    vector<vector<int>> edges_omp(n_rounds);
    vector<vector<bool>> results_omp(n_rounds);

    atomic<int> progress(0);

#pragma omp parallel for
    for (int i_round = 0; i_round < n_rounds; ++i_round) {
        int tid = omp_get_thread_num();
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(tid + seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        // sample nodes
        vector<int> sampled_nodes;
        for (int i = 0; i < n_nodes; ++i) {
            if (dist(rng) < alphas[i]) {
                sampled_nodes.push_back(i);
            }
        }
        auto num_sampled_nodes = sampled_nodes.size();
        if (num_sampled_nodes < 2) {
            progress++;
            if (tid == 0) display_progress(progress, n_rounds);
            continue;
        }

        // exhaustive binding is used: we do binding until all pairs are used
        auto random_e = dist(rng);
        for (int i = 0; i < num_sampled_nodes; ++i) {
            for (int j = i + 1; j < num_sampled_nodes; ++j) {
                auto p_ij = edge_probs[sampled_nodes[i]][sampled_nodes[j]];
                int edge_ij = sampled_nodes[i] * n_nodes + sampled_nodes[j];
                edges_omp[tid].push_back(edge_ij);
                results_omp[tid].push_back(random_e < p_ij);
            }
        }
        progress++;
        if (tid == 0) display_progress(progress, n_rounds);
    }
    display_progress(progress, n_rounds);
    cout << endl;

    // merge the results
    // create an n_nodes by n_nodes matrix to store the results, with default value -1
    cout << "Merging results" << endl;
    std::vector<std::vector<int8_t>> results(n_nodes, std::vector<int8_t>(n_nodes, -1));
    for (int i = 0; i < n_rounds; ++i) {
#pragma omp parallel for
        for (int j = 0; j < edges_omp[i].size(); ++j) {
            int vi = edges_omp[i][j] / n_nodes;
            int vj = edges_omp[i][j] % n_nodes;
            if (results[vi][vj] == -1) {
                results[vi][vj] = results_omp[i][j];
                results[vj][vi] = results_omp[i][j];
            }
        }
    }

    // sample the remaining pairs with independent sampling
    cout << "Sampling the remaining pairs" << endl;

    vector<vector<int>> edges_remain_omp(omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i) {
        for (int j = i + 1; j < n_nodes; ++j) {
            int tid = omp_get_thread_num();
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            mt19937 rng(tid + seed);
            uniform_real_distribution<double> dist(0.0, 1.0);
            if (results[i][j] == 1) {
                edges_remain_omp[tid].push_back(i * n_nodes + j);
            } else if (results[i][j] == -1) {  // sampling the remaining pairs with independent sampling
                auto random_e = dist(rng);
                if (random_e < edge_probs[i][j]) {
                    edges_remain_omp[tid].push_back(i * n_nodes + j);
                }
            }
        }
    }

    // Merge thread-local vectors into main vector
    for (const auto& edges_remain : edges_remain_omp) {
        edges.insert(edges.end(), edges_remain.begin(), edges_remain.end());
    }

    return edges;
}

vector<int> generate_edges_no_parallel(int n_nodes, vector<vector<double>> edge_probs, vector<double> alphas, int n_rounds) {
    // initialize an empty edge list
    vector<int> edges;
    int total_size = n_nodes * (n_nodes - 1) / 2;

    // vector<vector<int>> edges_omp(n_rounds);
    // vector<vector<bool>> results_omp(n_rounds);

    atomic<int> progress(0);
    // std::vector<std::mutex> res_mutexes(n_nodes * n_nodes - 1);
    std::vector<std::vector<int8_t>> results(n_nodes, std::vector<int8_t>(n_nodes, -1));

    for (int i_round = 0; i_round < n_rounds; ++i_round) {
        // int tid = omp_get_thread_num();
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // mt19937 rng(tid + seed);
        mt19937 rng(seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        // sample nodes
        vector<int> sampled_nodes;
        for (int i = 0; i < n_nodes; ++i) {
            if (dist(rng) < alphas[i]) {
                sampled_nodes.push_back(i);
            }
        }
        auto num_sampled_nodes = sampled_nodes.size();
        if (num_sampled_nodes < 2) {
            progress++;
            // if (tid == 0) display_progress(progress, n_rounds);
            display_progress(progress, n_rounds);
            continue;
        }

        // exhaustive binding is used: we do binding until all pairs are used
        auto random_e = dist(rng);
        vector<pair<int, int>> sampled_pairs;
        for (int i = 0; i < num_sampled_nodes; ++i) {
            int node_i = sampled_nodes[i];
            for (int j = i + 1; j < num_sampled_nodes; ++j) {
                int node_j = sampled_nodes[j];
                sampled_pairs.push_back(make_pair(node_i, node_j));
                int edge_ij = node_i * n_nodes + node_j;
                // res_mutexes[edge_ij].lock();
            }
        }
        // for each pair
        // #pragma omp parallel for
        for (auto i_pair = 0; i_pair < sampled_pairs.size(); ++i_pair) {
            auto node_i = sampled_pairs[i_pair].first;
            auto node_j = sampled_pairs[i_pair].second;
            auto p_ij = edge_probs[node_i][node_j];
            int edge_ij = node_i * n_nodes + node_j;
            bool success = random_e < p_ij;
            if (results[node_i][node_j] == -1) {
                results[node_i][node_j] = success;
                results[node_j][node_i] = success;
            }
            // res_mutexes[edge_ij].unlock();
        }

        progress++;
        display_progress(progress, n_rounds);
    }
    display_progress(progress, n_rounds);
    cout << endl;

    // sample the remaining pairs with independent sampling
    cout << "Sampling the remaining pairs" << endl;

    vector<vector<int>> edges_remain_omp(omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i) {
        for (int j = i + 1; j < n_nodes; ++j) {
            int tid = omp_get_thread_num();
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            mt19937 rng(tid + seed);
            uniform_real_distribution<double> dist(0.0, 1.0);
            if (results[i][j] == 1) {
                edges_remain_omp[tid].push_back(i * n_nodes + j);
            } else if (results[i][j] == -1) {  // sampling the remaining pairs with independent sampling
                auto random_e = dist(rng);
                if (random_e < edge_probs[i][j]) {
                    edges_remain_omp[tid].push_back(i * n_nodes + j);
                }
            }
        }
    }

    // Merge thread-local vectors into main vector
    for (const auto& edges_remain : edges_remain_omp) {
        edges.insert(edges.end(), edges_remain.begin(), edges_remain.end());
    }

    return edges;
}

// main function
int main(int argc, char const* argv[]) {
    // read the arguments
    // argv[1]: input filename
    // argv[2]: output filename
    // argv[3]: number of generated graphs
    // argv[4]: parallel or not

    // read the file name from argv into a string
    if (argc != 5) {
        cout << "Usage: " << argv[0]
             << " <filename_in>"
             << " <filename_out>"
             << " <n_graphs>"
             << " <parallel or not>"
             << endl;
        return 1;
    }
    string filename_in = argv[1];
    string filename_out = argv[2];
    int n_graphs = atoi(argv[3]);
    bool parallel = atoi(argv[4]);

    // read the input file
    // first line: n_blocks, n_round
    // each line after that: degree, number of nodes with this degree, alpha of this degree
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
    int n_nodes = total_nodes;

    // generate n_graphs graphs
    for (size_t i_graph = 0; i_graph < n_graphs; i_graph++) {
        // generate edge using binding
        auto start = std::chrono::steady_clock::now();  // Start the timer
        vector<int> edges;
        if (parallel) {
            edges = generate_edges(n_nodes, edge_probs, alphas, n_rounds);
        } else {
            edges = generate_edges_no_parallel(n_nodes, edge_probs, alphas, n_rounds);
        }
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
