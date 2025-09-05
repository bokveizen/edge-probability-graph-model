// include
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

template <typename T>
double get_pearson_correlation_coefficient(const std::vector<T>& x, const std::vector<T>& y) {
    size_t n = x.size();
    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);

    std::vector<double> x_squared(x.size());
    std::transform(x.begin(), x.end(), x_squared.begin(), [](T a) { return a * a; });
    double sum_x_squared = std::accumulate(x_squared.begin(), x_squared.end(), 0.0);

    std::vector<double> y_squared(y.size());
    std::transform(y.begin(), y.end(), y_squared.begin(), [](T a) { return a * a; });
    double sum_y_squared = std::accumulate(y_squared.begin(), y_squared.end(), 0.0);

    std::vector<double> xy(x.size());
    std::transform(x.begin(), x.end(), y.begin(), xy.begin(), std::multiplies<double>());
    double sum_xy = std::accumulate(xy.begin(), xy.end(), 0.0);

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x_squared - sum_x * sum_x) * (n * sum_y_squared - sum_y * sum_y));

    return numerator / denominator;
}

// main function
int main(int argc, char const* argv[]) {
    // read the arguments
    // argv[1]: input filename
    // argv[2]: output filename
    // argv[3]: skip the first line or not

    // read the file name from argv into a string
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <input file> <output file> <skip 1st line>" << endl;
        return 1;
    }
    string input_file_name = argv[1];
    string output_file_name = argv[2];
    string skip_first_line = argv[3];
    bool skip = false;
    if (skip_first_line == "1") {
        cout << "skip the first line" << endl;
        skip = true;
    } else if (skip_first_line == "0") {
        cout << "do not skip the first line" << endl;
    } else {
        // invalid input
        cout << "Usage: " << argv[0] << " <input file> <output file> <skip 1st line: 1/0>" << endl;
        return 1;
    }

    // open the input file
    ifstream input_file(input_file_name);
    if (!input_file.is_open()) {
        cout << "Error: cannot open file " << input_file_name << endl;
        return 1;
    }

    // read each line of the input file
    // each line is an edge, i.e., a pair of nodes
    // store the nodes and edges in a vector

    // skip the first line if needed
    if (skip) {
        string line;
        getline(input_file, line);
    }

    vector<int> nodes;
    vector<pair<int, int>> edges;
    int n_nodes = 0;
    int n_edges = 0;
    int node1, node2;

    // read the rest of the file
    while (input_file >> node1 >> node2) {
        // check if the nodes are already in the vector
        // if not, add them to the vector
        if (find(nodes.begin(), nodes.end(), node1) == nodes.end()) {
            nodes.push_back(node1);
            n_nodes++;
        }
        if (find(nodes.begin(), nodes.end(), node2) == nodes.end()) {
            nodes.push_back(node2);
            n_nodes++;
        }
        // add the edge to the vector
        edges.push_back(make_pair(node1, node2));
        n_edges++;
    }
    input_file.close();

    // find the max node
    int v_max = 0;
    for (auto node : nodes) {
        if (node > v_max) {
            v_max = node;
        }
    }
    vector<int> node_to_index(v_max + 1, 0);
    for (int i = 0; i < n_nodes; i++) {
        node_to_index[nodes[i]] = i;
    }

    // print the number of nodes and edges
    cout << "Number of nodes: " << n_nodes << endl;
    cout << "Number of edges: " << n_edges << endl;

    // analyze the graph to obtain the following information
    // number of nodes, number of edges, number of triads, number of triangles
    // global clustering coefficient, average local clustering coefficient
    // the pearson correlation coefficient between the degree and clustering coefficient
    // degree assortativity

    // first collect the degree of each node
    vector<int> degree(n_nodes, 0);
    vector<vector<int>> adjList(n_nodes);
    for (int i = 0; i < n_edges; i++) {
        // map the nodes to indices
        edges[i].first = node_to_index[edges[i].first];
        edges[i].second = node_to_index[edges[i].second];
        adjList[edges[i].first].push_back(edges[i].second);
        adjList[edges[i].second].push_back(edges[i].first);
        degree[edges[i].first] += 1;
        degree[edges[i].second] += 1;
    }
    cout << "degrees done" << endl;

    // Sort the adjacency list
    for (int i = 0; i < n_nodes; i++) {
        sort(adjList[i].begin(), adjList[i].end());
    }
    cout << "sorting adj done" << endl;

    // compute the number of triads using the degrees
    int n_triads = 0;
    for (auto deg : degree) {
        n_triads += deg * (deg - 1) / 2;
    }
    cout << "triads done" << endl;

    // Count triangles, both the total number and the number of triangles each node is involved in
    vector<int> n_triangles_per_node(n_nodes, 0);
#pragma omp parallel for
    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < adjList[i].size(); j++) {
            for (int k = j + 1; k < adjList[i].size(); k++) {
                if (binary_search(adjList[adjList[i][j]].begin(), adjList[adjList[i][j]].end(), adjList[i][k])) {
                    n_triangles_per_node[i]++;
                }
            }
        }
    }
    int n_triangles = 0;
    for (auto n : n_triangles_per_node) {
        n_triangles += n;
    }
    int n_open_triangles = n_triads - n_triangles;
    cout << "triangles done" << endl;

    // compute the global clustering coefficient
    double global_clustering_coefficient = (double)n_triangles / n_triads;
    cout << "global clustering coefficient done" << endl;

    // compute the local clustering coefficient for each node
    vector<double> local_clustering_coefficient(n_nodes, 0.0);
    for (int i = 0; i < n_nodes; i++) {
        if (degree[i] > 1) {
            local_clustering_coefficient[i] = 2.0 * n_triangles_per_node[i] / (degree[i] * (degree[i] - 1));
        }
    }

    // compute the average local clustering coefficient
    double average_local_clustering_coefficient = 0.0;
    for (auto lcc : local_clustering_coefficient) {
        average_local_clustering_coefficient += lcc;
    }
    average_local_clustering_coefficient /= n_nodes;
    cout << "local clustering coefficient done" << endl;

    // compute the pearson correlation coefficient between the degree and clustering coefficient
    // convert degree to double
    vector<double> degree_double(n_nodes, 0.0);
    for (int i = 0; i < n_nodes; i++) {
        degree_double[i] = degree[i];
    }
    double pearson_correlation_coefficient = get_pearson_correlation_coefficient(degree_double, local_clustering_coefficient);
    cout << "pearson correlation coefficient done" << endl;

    // compute the degree assortativity
    // first collect the degrees in two vectors and then compute the pearson correlation coefficient between the two vectors
    vector<int> degree1;
    vector<int> degree2;
    for (int i = 0; i < n_edges; i++) {
        degree1.push_back(degree_double[edges[i].first]);
        degree1.push_back(degree_double[edges[i].second]);
        degree2.push_back(degree_double[edges[i].second]);
        degree2.push_back(degree_double[edges[i].first]);
    }
    double degree_assortativity = get_pearson_correlation_coefficient(degree1, degree2);
    cout << "degree assortativity done" << endl;

    // print the results to the output file (appending) as well as the screen
    ofstream output_file(output_file_name, ios::app);
    if (!output_file.is_open()) {
        cout << "Error: cannot open file " << output_file_name << endl;
        return 1;
    }
    // in the output file, print the results in a single line
    // separated by a space
    // the order is: number of nodes, number of edges, number of triads, number of open triangles, number of triangles
    // global clustering coefficient, average local clustering coefficient, pearson correlation coefficient, degree assortativity
    output_file << n_nodes << " " << n_edges << " " << n_triads << " " << n_open_triangles << " " << n_triangles << " "
                << global_clustering_coefficient << " " << average_local_clustering_coefficient << " "
                << pearson_correlation_coefficient << " " << degree_assortativity << endl;
    output_file.close();

    cout << "Number of nodes: " << n_nodes << endl;
    cout << "Number of edges: " << n_edges << endl;
    cout << "Number of triads: " << n_triads << endl;
    cout << "Number of triangles: " << n_triangles << endl;
    cout << "Global clustering coefficient: " << global_clustering_coefficient << endl;
    cout << "Average local clustering coefficient: " << average_local_clustering_coefficient << endl;
    cout << "Pearson correlation coefficient: " << pearson_correlation_coefficient << endl;
    cout << "Degree assortativity: " << degree_assortativity << endl;

    return 0;
}