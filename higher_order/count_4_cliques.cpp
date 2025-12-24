#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <omp.h>

// Function to read the edge list from a file and build the adjacency list
bool readEdgeList(const std::string& filename, std::vector<std::vector<int>>& adjList, int& numNodes) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open the file " << filename << std::endl;
        return false;
    }

    int u, v;
    int maxNode = -1;
    std::vector<std::pair<int, int>> edges;

    // Read edges and determine the maximum node id
    while (infile >> u >> v) {
        edges.emplace_back(u, v);
        if (u > maxNode) maxNode = u;
        if (v > maxNode) maxNode = v;
    }

    infile.close();

    // Assuming nodes are labeled from 0 to maxNode
    numNodes = maxNode + 1;
    adjList.assign(numNodes, std::vector<int>());

    // Populate adjacency list
    for (const auto& edge : edges) {
        u = edge.first;
        v = edge.second;
        adjList[u].push_back(v);
        adjList[v].push_back(u); // Since the graph is undirected
    }

    // Sort the adjacency lists for efficient searching
    for (int i = 0; i < numNodes; ++i) {
        std::sort(adjList[i].begin(), adjList[i].end());
    }

    return true;
}

// Function to check if two nodes are connected using binary search
bool isConnected(const std::vector<std::vector<int>>& adjList, int u, int v) {
    const auto& neighbors = adjList[u];
    return std::binary_search(neighbors.begin(), neighbors.end(), v);
}

int main(int argc, char* argv[]) {
    // Check if the filename is provided as a command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <edge_list_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1]; // Get the filename from arguments
    std::vector<std::vector<int>> adjList;
    int numNodes = 0;

    // Read the edge list and build the adjacency list
    if (!readEdgeList(filename, adjList, numNodes)) {
        return 1;
    }

    std::cout << "Number of nodes: " << numNodes << std::endl;

    // Set number of threads (use all available by default)
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;

    long long fourCliques = 0;

    // Enumerate all unique 4-node combinations (u < v < w < x)
    // Parallelize the outer loop for better load balancing
    #pragma omp parallel for schedule(dynamic) reduction(+:fourCliques)
    for (int u = 0; u < numNodes - 3; ++u) {
        for (int v = u + 1; v < numNodes - 2; ++v) {
            // Check if u and v are connected
            if (!isConnected(adjList, u, v)) continue;

            for (int w = v + 1; w < numNodes - 1; ++w) {
                // Check if u and w, and v and w are connected
                if (!isConnected(adjList, u, w) || !isConnected(adjList, v, w)) continue;

                for (int x = w + 1; x < numNodes; ++x) {
                    // Check if x is connected to u, v, and w
                    if (isConnected(adjList, u, x) &&
                        isConnected(adjList, v, x) &&
                        isConnected(adjList, w, x)) {
                        fourCliques++;
                    }
                }
            }
        }
    }

    std::cout << "Number of 4-cliques: " << fourCliques << std::endl;

    return 0;
}