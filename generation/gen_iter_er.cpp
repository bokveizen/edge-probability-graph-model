// underlying: Erdos Renyi
// binding: iterative exhaustive
// input: (number of nodes, density, binding strength, number of rounds)

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
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

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
// prob: a (uniform) edge probability
// alpha: a (uniform) binding strength
// n_rounds: number of total rounds
// output //
// a list of edges
vector<int> generate_edges(int n_nodes, double prob, double alpha, int n_rounds) {
    // initialize an empty edge list
    // vector<pair<int, int>> edges;
    vector<int> edges;
    int total_size = n_nodes * (n_nodes - 1) / 2;

    // vector<vector<pair<int, int>>> edges_omp(omp_get_max_threads());
    // vector<vector<int>> edges_omp(omp_get_max_threads());
    // vector<vector<bool>> results_omp(omp_get_max_threads());

    vector<vector<int>> edges_omp(n_rounds);
    vector<vector<bool>> results_omp(n_rounds);

    atomic<int> progress(0);

#pragma omp parallel for
    for (int i_round = 0; i_round < n_rounds; ++i_round) {
        int tid = omp_get_thread_num();
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(tid + seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        // sample nodes from "nodes", where each node is sampled with probability alpha
        vector<int> sampled_nodes;
        for (int i = 0; i < n_nodes; ++i) {
            if (dist(rng) < alpha) {
                sampled_nodes.push_back(i);
            }
        }
        auto num_sampled_nodes = sampled_nodes.size();
        // if the number of sampled nodes is less than 2, then we cannot sample any edge and we can skip this round
        if (num_sampled_nodes < 2) {
            progress++;
            if (tid == 0) display_progress(progress, n_rounds);
            continue;
        }

        // exhaustive binding is used: we do binding until all pairs are used
        auto random_e = dist(rng);
        for (int i = 0; i < num_sampled_nodes; ++i) {
            for (int j = i + 1; j < num_sampled_nodes; ++j) {
                // edges_omp[tid].push_back({sampled_nodes[i], sampled_nodes[j]});
                int ij = sampled_nodes[i] * n_nodes + sampled_nodes[j];
                // edges_omp[tid].push_back(ij);
                // results_omp[tid].push_back(random_e < prob);
                edges_omp[i_round].push_back(ij);
                results_omp[i_round].push_back(random_e < prob);
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
    // for (int i = 0; i < omp_get_max_threads(); ++i) {
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
    // mutex mtx_edges;

    vector<vector<int>> edges_remain_omp(omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i) {
        for (int j = i + 1; j < n_nodes; ++j) {
            int tid = omp_get_thread_num();
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            mt19937 rng(tid + seed);
            uniform_real_distribution<double> dist(0.0, 1.0);
            if (results[i][j] == 1) {                
                // mtx_edges.lock();
                // edges.push_back(i * n_nodes + j);
                // mtx_edges.unlock();
                edges_remain_omp[tid].push_back(i * n_nodes + j);
            } else if (results[i][j] == -1) {  // sampling the remaining pairs with independent sampling
                auto random_e = dist(rng);
                if (random_e < prob) {                    
                    // mtx_edges.lock();
                    // edges.push_back(i * n_nodes + j);
                    // mtx_edges.unlock();
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

vector<int> generate_edges_no_parallel(int n_nodes, double prob, double alpha, int n_rounds) {
    // initialize an empty edge list
    // vector<pair<int, int>> edges;
    vector<int> edges;
    int total_size = n_nodes * (n_nodes - 1) / 2;

    // vector<vector<pair<int, int>>> edges_omp(omp_get_max_threads());
    // vector<vector<int>> edges_omp(omp_get_max_threads());
    // vector<vector<bool>> results_omp(omp_get_max_threads());

    // vector<vector<int>> edges_omp(n_rounds);
    // vector<vector<bool>> results_omp(n_rounds);

    atomic<int> progress(0);

    std::vector<std::vector<int8_t>> results(n_nodes, std::vector<int8_t>(n_nodes, -1));

    for (int i_round = 0; i_round < n_rounds; ++i_round) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        // sample nodes from "nodes", where each node is sampled with probability alpha
        vector<int> sampled_nodes;
        for (int i = 0; i < n_nodes; ++i) {
            if (dist(rng) < alpha) {
                sampled_nodes.push_back(i);
            }
        }
        auto num_sampled_nodes = sampled_nodes.size();
        // if the number of sampled nodes is less than 2, then we cannot sample any edge and we can skip this round
        if (num_sampled_nodes < 2) {
            progress++;
            display_progress(progress, n_rounds);
            continue;
        }

        // exhaustive binding is used: we do binding until all pairs are used
        auto random_e = dist(rng);
        for (int i = 0; i < num_sampled_nodes; ++i) {
            for (int j = i + 1; j < num_sampled_nodes; ++j) {
                int vi = sampled_nodes[i];
                int vj = sampled_nodes[j];
                // edges_omp[tid].push_back({sampled_nodes[i], sampled_nodes[j]});
                int ij = sampled_nodes[i] * n_nodes + sampled_nodes[j];                                
                bool success = random_e < prob;
                if (results[vi][vj] == -1) {
                    results[vi][vj] = success;
                    results[vj][vi] = success;
                }
            }
        }
        progress++;
        display_progress(progress, n_rounds);
    }
    display_progress(progress, n_rounds);
    cout << endl;

    // sample the remaining pairs with independent sampling
    cout << "Sampling the remaining pairs" << endl;
    // mutex mtx_edges;

    vector<vector<int>> edges_remain_omp(omp_get_max_threads());
    #pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i) {
        for (int j = i + 1; j < n_nodes; ++j) {
            int tid = omp_get_thread_num();
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            mt19937 rng(tid + seed);
            uniform_real_distribution<double> dist(0.0, 1.0);
            if (results[i][j] == 1) {
                // mtx_edges.lock();
                // edges.push_back(i * n_nodes + j);
                // mtx_edges.unlock();
                edges_remain_omp[tid].push_back(i * n_nodes + j);
            } else if (results[i][j] == -1) {  // sampling the remaining pairs with independent sampling
                auto random_e = dist(rng);
                if (random_e < prob) {
                    // mtx_edges.lock();
                    // edges.push_back(i * n_nodes + j);
                    // mtx_edges.unlock();
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
             << " <n_output>"
             << " <parallel or not>"
             << endl;
        return 1;
    }
    string filename_in = argv[1];
    string filename_out = argv[2];
    int n_graphs = atoi(argv[3]);
    bool parallel = atoi(argv[4]);

    // read the input file: number of nodes, density, binding strength, number of rounds
    ifstream infile;
    infile.open(filename_in);
    int n_nodes, n_rounds;
    double p, alpha;    
    infile >> n_nodes >> p >> alpha >> n_rounds;
    infile.close();

    // print the arguments
    cout << "Number of nodes: " << n_nodes << endl;
    cout << "Density: " << p << endl;
    cout << "Binding strength alpha: " << alpha << endl;
    cout << "Number of rounds: " << n_rounds << endl;
    cout << "Output file: " << filename_out << endl;
    cout << "Number of generated graphs: " << n_graphs << endl;

    // generate n_graphs graphs
    for (auto i_graph = 0; i_graph < n_graphs; i_graph++) {
        // generate edge using binding
        auto start = std::chrono::steady_clock::now();  // Start the timer
        
        vector<int> edges;
        if (parallel) {
            edges = generate_edges(n_nodes, p, alpha, n_rounds);
        } else {
            edges = generate_edges_no_parallel(n_nodes, p, alpha, n_rounds);
        }

        auto end = std::chrono::steady_clock::now();                                         // Stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  // Calculate the duration in milliseconds

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
        for (auto i = 0; i < edges.size(); ++i) {
            int vi = edges[i] / n_nodes;
            int vj = edges[i] % n_nodes;
            nodes_in_edges.push_back(vi);
            nodes_in_edges.push_back(vj);
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
            int vi = edges[i] / n_nodes;
            int vj = edges[i] % n_nodes;
            // outfile << edges[i].first << " " << edges[i].second << endl;
            outfile << vi << " " << vj << endl;
        }

        auto end_write = std::chrono::steady_clock::now();                                                     // Stop the timer
        auto duration_write = std::chrono::duration_cast<std::chrono::milliseconds>(end_write - start_write);  // Calculate the duration in milliseconds
        double time_write = duration_write.count() / 1000.0;

        outfile.close();
    }
    cout << "Program done" << endl;

    return 0;
}
