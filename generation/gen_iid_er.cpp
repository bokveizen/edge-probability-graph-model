// underlying: Erdos Renyi
// binding: iterative exhaustive
// input: (number of nodes, density, binding strength, number of rounds)

#include <omp.h>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/pow.hpp>
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

typedef boost::multiprecision::cpp_dec_float_50 float50;  // 50 digits of precision

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
// vector<int> generate_edges(int n_nodes, double prob, double alpha, int n_rounds) {
vector<int> generate_edges(int n_nodes, float50 prob, float50 alpha, int n_rounds) {
    // initialize an empty edge list    
    // vector<pair<int, int>> edges;
    vector<int> edges;

    // in total "n_rounds" rounds
    // we first compute the number of rounds with insertion
    // then for each "success" round, we sample nodes in "nodes", where each node is sampled with probabilitiy "alpha"
    // then we add a clique between the sampled nodes

    // vector<vector<pair<int, int>>> edges_omp(omp_get_max_threads());
    vector<vector<int>> edges_omp(omp_get_max_threads());
    atomic<int> progress(0);

    // compute the number of rounds with insertion
    int n_rounds_gen;
    if (prob == 1.0) {
        n_rounds_gen = n_rounds;
    } else {
        // binomial distribution (n_rounds, prob)
        // random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(seed);
        // case prob to double
        double prob_double = prob.convert_to<double>();
        binomial_distribution<int> dist_binom(n_rounds, prob_double);
        n_rounds_gen = dist_binom(rng);
    }

#pragma omp parallel for
    for (int i = 0; i < n_rounds_gen; ++i) {
        int tid = omp_get_thread_num();

        // random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng(tid + seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        // sample nodes
        vector<int> sampled_nodes;
        for (int j = 0; j < n_nodes; ++j) {
            double random_v = dist(rng);
            if (random_v < alpha) {
                sampled_nodes.push_back(j);
            }
        }

        // if the number of sampled nodes is less than 2, skip this round
        if (sampled_nodes.size() < 2) {
            progress++;
            if (tid == 0) {
                display_progress(progress, n_rounds_gen);
            }
            continue;
        }

        // sample edges
        for (int j = 0; j < sampled_nodes.size() - 1; ++j) {
            for (int k = j + 1; k < sampled_nodes.size(); ++k) {
                int edge_jk = sampled_nodes[j] * n_nodes + sampled_nodes[k];
                // edges_omp[tid].push_back(make_pair(sampled_nodes[j], sampled_nodes[k]));
                edges_omp[tid].push_back(edge_jk);
            }
        }

        progress++;
        if (tid == 0) {
            display_progress(progress, n_rounds_gen);
        }
    }
    cout << endl;

    // merge the edges
    for (const auto& e : edges_omp) {
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

    // read the input file: number of nodes, density, binding strength, number of rounds
    ifstream infile;
    infile.open(filename_in);
    int n_nodes, n_rounds;
    float50 p, alpha, p_round;
    infile >> n_nodes >> p >> alpha >> n_rounds >> p_round;
    infile.close();

    // print the arguments
    cout << "Number of nodes: " << n_nodes << endl;
    cout << "Density: " << p << endl;
    cout << "Binding strength alpha: " << alpha << endl;
    cout << "Number of rounds: " << n_rounds << endl;
    cout << "Output file: " << filename_out << endl;
    cout << "Number of generated graphs: " << n_graphs << endl;

    // then, compute the number of success probability in each round
    float50 n_rounds_float50 = n_rounds;
    // float50 pow_res = boost::multiprecision::pow(1.0 - p, 1.0 / n_rounds_float50);
    // float50 p_round_compute = (1.0 - pow(1.0 - p, 1.0 / n_rounds_float50)) / (alpha * alpha);
    // float50 p_round = (1.0 - pow_res) / (alpha * alpha);    
    float50 p_remain = 0.0;

    // cout << "p: " << p << endl;
    // cout << "p_round_compute: " << p_round_compute << endl;
    // cout << "p_round_direct: " << p_round_direct << endl;
    // cout << "p_round: " << p_round << endl;
    // return 0;


    bool has_remain = false;
    if (p_round > 1.0) {
        p_round = 1.0;
        float50 p_max = 1.0 - pow(1.0 - alpha * alpha, n_rounds);
        p_remain = 1.0 - (1.0 - p) / (1.0 - p_max);
        has_remain = true;
    }
    cout << "p: " << p << endl;
    cout << "p_round: " << p_round << endl;
    // return 0;
    // cout << "p_remain: " << p_remain << endl;

    // a vector of all possible pairs
    // vector<pair<int, int>> pairs;
    // vector<int> pairs;
    // if (has_remain) {
    //     for (int v1 = 0; v1 < n_nodes; ++v1) {
    //         for (int v2 = v1 + 1; v2 < n_nodes; ++v2) {                
    //             // pairs.push_back({v1, v2});
    //             pairs.push_back(v1 * n_nodes + v2);
    //         }
    //     }
    // }

    // Create a vector of vectors, one for each thread
    std::vector<std::vector<int>> pairs_omp(omp_get_max_threads());

    #pragma omp parallel for
    for (int v1 = 0; v1 < n_nodes; ++v1) {
        int tid = omp_get_thread_num();
        for (int v2 = v1 + 1; v2 < n_nodes; ++v2) {
            pairs_omp[tid].push_back(v1 * n_nodes + v2);
        }
    }

    // Merge thread-local vectors into main vector
    std::vector<int> pairs;
    for (const auto& p : pairs_omp) {
        pairs.insert(pairs.end(), p.begin(), p.end());
    }

    // generate n_graphs graphs
    for (auto i_graph = 0; i_graph < n_graphs; i_graph++) {
        // generate edge using binding
        auto start = std::chrono::steady_clock::now();  // Start the timer

        // vector<pair<int, int>> edges = generate_edges(n_nodes, p_round, alpha, n_rounds);
        vector<int> edges = generate_edges(n_nodes, p_round, alpha, n_rounds);

        // add the remaining edges if has_remain
        if (has_remain) {
            // add each pair with probability p_remain
            // random number generator
            uniform_real_distribution<double> dist(0.0, 1.0);
            // vector<vector<pair<int, int>>> edges_remain_omp(omp_get_max_threads());
            vector<vector<int>> edges_remain_omp(omp_get_max_threads());
#pragma omp parallel for
            for (auto i = 0; i < pairs.size(); ++i) {
                int tid = omp_get_thread_num();
                auto seed = std::chrono::system_clock::now().time_since_epoch().count();
                mt19937 rng(tid + seed);
                double random_e = dist(rng);
                if (random_e < p_remain) {
                    edges_remain_omp[tid].push_back(pairs[i]);
                }
            }
            for (const auto& e : edges_remain_omp) {
                edges.insert(edges.end(), e.begin(), e.end());
            }
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
            // nodes_in_edges.push_back(edges[i].first);
            // nodes_in_edges.push_back(edges[i].second);
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
            // outfile << edges[i].first << " " << edges[i].second << endl;
            int vi = edges[i] / n_nodes;
            int vj = edges[i] % n_nodes;
            outfile << vi << " " << vj << endl;
        }

        auto end_write = std::chrono::steady_clock::now();                                                     // Stop the timer
        auto duration_write = std::chrono::duration_cast<std::chrono::milliseconds>(end_write - start_write);  // Calculate the duration in milliseconds
        double time_write = duration_write.count() / 1000.0;

        outfile.close();
    }
    return 0;
}
