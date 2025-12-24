#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using std::size_t;
using std::uint32_t;
using std::uint64_t;

// ---------- utilities ----------
inline uint64_t encode_pair_u64(uint64_t u, uint64_t v, uint64_t n_nodes) {
    if (u > v) std::swap(u, v);
    return u * n_nodes + v;
}
inline std::pair<uint64_t,uint64_t> decode_pair_u64(uint64_t key, uint64_t n_nodes) {
    return { key / n_nodes, key % n_nodes };
}
inline double q_from_p(double p, uint32_t R) {
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1.0;
    return -std::expm1(std::log1p(-p) / static_cast<double>(R));
}

// nCk in uint64 for k<=63 (works for typical KR settings)
static inline uint64_t nCk_u64(uint32_t n, uint32_t k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    k = std::min(k, n - k);
    __int128 res = 1;
    for (uint32_t i = 1; i <= k; ++i) {
        res = (res * static_cast<__int128>(n - k + i)) / static_cast<__int128>(i);
    }
    return static_cast<uint64_t>(res);
}

// combinadic unranking: rank in [0, C(k,w)-1] -> k-bit mask with exactly w ones
static inline uint64_t unrank_combination_bitmask(uint32_t k, uint32_t w, uint64_t rank) {
    uint64_t mask = 0ULL;
    uint32_t rem = w;
    for (int pos = static_cast<int>(k) - 1; pos >= 0 && rem > 0; --pos) {
        uint64_t c = nCk_u64(static_cast<uint32_t>(pos), rem - 1);
        if (rank >= c) {
            rank -= c;
        } else {
            mask |= (1ULL << pos);
            rem--;
        }
    }
    return mask;
}

// Floyd sampling: K distinct ints from [0..N-1] in O(K)
static inline std::vector<uint64_t> sample_unique_ranks(uint64_t N, uint64_t K, std::mt19937_64 &rng) {
    std::vector<uint64_t> out;
    out.reserve(static_cast<size_t>(K));
    if (K == 0) return out;
    if (K > N) K = N;

    std::unordered_set<uint64_t> S;
    S.reserve(static_cast<size_t>(K * 1.3));

    for (uint64_t j = N - K; j < N; ++j) {
        std::uniform_int_distribution<uint64_t> dist(0, j);
        uint64_t t = dist(rng);
        if (S.find(t) != S.end()) S.insert(j);
        else S.insert(t);
    }
    out.assign(S.begin(), S.end());
    return out;
}

static inline std::vector<uint64_t> sample_nodes_fixed_weight(uint32_t k, uint32_t w, uint64_t K, std::mt19937_64 &rng) {
    uint64_t Nw = nCk_u64(k, w);
    if (Nw == 0 || K == 0) return {};
    if (K > Nw) K = Nw;

    std::vector<uint64_t> ranks = sample_unique_ranks(Nw, K, rng);
    std::vector<uint64_t> nodes;
    nodes.reserve(ranks.size());
    for (uint64_t r : ranks) nodes.push_back(unrank_combination_bitmask(k, w, r));
    return nodes;
}

// ---------- output types (compressed) ----------
struct FullClique {
    uint32_t w;
    std::vector<uint64_t> nodes;   // all edges inside are present
};
struct FullBiclique {
    uint32_t w_left, w_right;
    std::vector<uint64_t> left, right; // all cross edges are present
};

// ---------- precompute r(wu,wv,d) under symmetric seed ----------
struct KRPrecompSym {
    uint32_t k = 0;
    uint32_t R = 0;

    // r_by_dist[wu][wv][d] for d in [0..k], invalid entries = -1
    std::vector<std::vector<std::vector<double>>> r_by_dist;

    // min/max over feasible d for each (wu,wv)
    std::vector<std::vector<double>> rmin;
    std::vector<std::vector<double>> rmax;

    double global_rmax = 0.0;
};

// Symmetric binary KR: p = a^{n00} b^{d} c^{n11}
// with: n11 = (wu+wv-d)/2 , n00 = k - wu - wv + n11, feasibility requires parity and bounds.
static inline KRPrecompSym build_precomp_symmetric_seed(
    uint32_t k, uint32_t R,
    double a, double b, double c,
    const std::vector<double> &g_w   // size k+1
) {
    KRPrecompSym T;
    T.k = k; T.R = R;
    T.r_by_dist.assign(k + 1, std::vector<std::vector<double>>(k + 1, std::vector<double>(k + 1, -1.0)));
    T.rmin.assign(k + 1, std::vector<double>(k + 1, 1.0));
    T.rmax.assign(k + 1, std::vector<double>(k + 1, 0.0));

    const double loga = std::log(a);
    const double logb = std::log(b);
    const double logc = std::log(c);

    for (uint32_t wu = 0; wu <= k; ++wu) {
        for (uint32_t wv = 0; wv <= k; ++wv) {
            double local_min = 1.0;
            double local_max = 0.0;

            // feasible d range with parity:
            // d = wu+wv-2*n11 => d has same parity as wu+wv
            uint32_t lo = std::abs((int)wu - (int)wv);
            uint32_t hi = std::min<uint32_t>(k, wu + wv);
            // also d <= (k-wu)+(k-wv) = 2k-wu-wv; but hi=k already covers typical;
            hi = std::min<uint32_t>(hi, 2*k - wu - wv);

            for (uint32_t d = lo; d <= hi; ++d) {
                if (((wu + wv - d) & 1U) != 0U) continue; // parity infeasible

                uint32_t n11 = (wu + wv - d) / 2;
                // n11 must be <= min(wu,wv) automatically satisfied by lo/hi, but keep safe:
                if (n11 > std::min(wu, wv)) continue;

                uint32_t n00 = k - wu - wv + n11;
                // n00 must be >=0; if negative, infeasible
                // (unsigned underflow check)
                if ((int32_t)n00 < 0) continue;

                double gw_u = g_w[wu], gw_v = g_w[wv];
                if (gw_u <= 0.0 || gw_v <= 0.0) {
                    T.r_by_dist[wu][wv][d] = 0.0;
                    local_min = 0.0;
                    continue;
                }

                double logp = n00 * loga + d * logb + n11 * logc;
                double p = (logp < -800.0) ? 0.0 : std::exp(logp);
                double q = q_from_p(p, R);

                double r = q / (gw_u * gw_v);
                if (r > 1.0) r = 1.0;

                T.r_by_dist[wu][wv][d] = r;
                local_min = std::min(local_min, r);
                local_max = std::max(local_max, r);
            }

            T.rmin[wu][wv] = local_min;
            T.rmax[wu][wv] = local_max;
            T.global_rmax = std::max(T.global_rmax, local_max);
        }
    }
    return T;
}

// ---------- main generator ----------
static inline void generate_edges_kr_pb_symmetric(
    uint32_t R, uint32_t k, uint64_t n_nodes,
    const std::vector<double> &g_w,
    const KRPrecompSym &T,
    std::vector<FullClique> &full_cliques,
    std::vector<FullBiclique> &full_bicliques,
    std::vector<uint64_t> &partial_edges
) {
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<FullClique>> clq_tls(n_threads);
    std::vector<std::vector<FullBiclique>> bic_tls(n_threads);
    std::vector<std::vector<uint64_t>> e_tls(n_threads);

    uint64_t seed_base = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );

    // Optional skip of threshold-inactive rounds: if s > global_rmax, no edges can be inserted.
    // Sample R_eff ~ Bin(R, global_rmax) and then s ~ Unif(0, global_rmax).
    const double rmax_global = T.global_rmax;
    const bool use_skip = (rmax_global < 1.0);

    uint32_t R_eff = R;
    if (use_skip) {
        std::mt19937_64 rng(seed_base ^ 0x9e3779b97f4a7c15ULL);
        std::binomial_distribution<uint32_t> dist_eff(R, rmax_global);
        R_eff = dist_eff(rng);
    }

    std::atomic<uint32_t> progress(0);
    std::mutex progress_mutex;
    auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 8)
    for (int ir = 0; ir < (int)R_eff; ++ir) {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(seed_base + 0xD1B54A32D192ED03ULL * (uint64_t)(tid + 1) + (uint64_t)ir);

        std::uniform_real_distribution<double> dist_s(0.0, use_skip ? rmax_global : 1.0);
        double s = dist_s(rng);

        // ---- group-wise sampling (by weight) ----
        std::vector<std::vector<uint64_t>> Vw(k + 1);
        std::vector<uint32_t> weights_present;
        weights_present.reserve(k + 1);

        for (uint32_t w = 0; w <= k; ++w) {
            double gw = g_w[w];
            if (gw <= 0.0) continue;

            uint64_t Nw = nCk_u64(k, w);
            if (Nw == 0) continue;

            std::binomial_distribution<uint64_t> dist_kw(Nw, gw);
            uint64_t Kw = dist_kw(rng);
            if (Kw == 0) continue;

            Vw[w] = sample_nodes_fixed_weight(k, w, Kw, rng);
            if (!Vw[w].empty()) weights_present.push_back(w);
        }

        if (weights_present.empty()) continue;

        // ---- configuration-wise processing using merged mismatches d = popcount(u^v) ----
        // Iterate only over weights that appear.
        for (size_t ai = 0; ai < weights_present.size(); ++ai) {
            uint32_t wu = weights_present[ai];
            auto &A = Vw[wu];

            // Within-group edges (wu == wv)
            if (A.size() >= 2) {
                double rmin = T.rmin[wu][wu];
                double rmax = T.rmax[wu][wu];

                if (s <= rmin) {
                    // full clique within A
                    clq_tls[tid].push_back(FullClique{wu, A});
                } else if (s <= rmax) {
                    // partial: check by distance d = popcount(u^v)
                    const auto &rt = T.r_by_dist[wu][wu];
                    for (size_t i = 0; i < A.size(); ++i) {
                        uint64_t u = A[i];
                        for (size_t j = i + 1; j < A.size(); ++j) {
                            uint64_t v = A[j];
                            uint32_t d = (uint32_t)__builtin_popcountll(u ^ v);
                            double r_uv = rt[d];
                            if (r_uv >= s) e_tls[tid].push_back(encode_pair_u64(u, v, n_nodes));
                        }
                    }
                }
            }

            // Cross-group edges (wu < wv)
            for (size_t bi = ai + 1; bi < weights_present.size(); ++bi) {
                uint32_t wv = weights_present[bi];
                auto &B = Vw[wv];

                double rmin = T.rmin[wu][wv];
                double rmax = T.rmax[wu][wv];

                if (s <= rmin) {
                    // full biclique A x B
                    bic_tls[tid].push_back(FullBiclique{wu, wv, A, B});
                } else if (s <= rmax) {
                    // partial
                    const auto &rt = T.r_by_dist[wu][wv];

                    // minor micro-opt: iterate smaller outer loop
                    if (A.size() <= B.size()) {
                        for (uint64_t u : A) {
                            for (uint64_t v : B) {
                                uint32_t d = (uint32_t)__builtin_popcountll(u ^ v);
                                double r_uv = rt[d];
                                if (r_uv >= s) e_tls[tid].push_back(encode_pair_u64(u, v, n_nodes));
                            }
                        }
                    } else {
                        for (uint64_t v : B) {
                            for (uint64_t u : A) {
                                uint32_t d = (uint32_t)__builtin_popcountll(u ^ v);
                                double r_uv = rt[d];
                                if (r_uv >= s) e_tls[tid].push_back(encode_pair_u64(u, v, n_nodes));
                            }
                        }
                    }
                }
            }
        }

        // progress (optional)
        uint32_t p = ++progress;
        if (tid == 0 && (p % std::max(1U, R_eff / 100U) == 0)) {
            std::lock_guard<std::mutex> lock(progress_mutex);
            double frac = (double)p / (double)R_eff;
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cerr << "\rKR rounds " << p << "/" << R_eff
                      << " (" << std::fixed << std::setprecision(1) << (100.0 * frac) << "%)"
                      << ", elapsed " << std::setprecision(1) << elapsed << "s" << std::flush;
        }
    }

    std::cerr << "\rKR rounds " << R_eff << "/" << R_eff << " (100.0%), done.            \n";

    // merge
    for (auto &v : clq_tls)
        full_cliques.insert(full_cliques.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    for (auto &v : bic_tls)
        full_bicliques.insert(full_bicliques.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    for (auto &v : e_tls)
        partial_edges.insert(partial_edges.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
}


// --------- helper: count zeros in k-bit representation (for reading alpha by #zeros) ----------
static inline uint32_t count_zeros_u64(uint64_t x, uint32_t k) {
    return k - (uint32_t)__builtin_popcountll(x & ((k == 64) ? ~0ULL : ((1ULL << k) - 1ULL)));
}

int main(int argc, char const *argv[]) {
    // CLI compatible with your old program:
    // argv[1]: input filename
    // argv[2]: output prefix
    // argv[3]: number of graphs
    // argv[4]: up_scale (adds to k)
    // argv[5]: expand (0/1) -- ignored here

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <filename_in> <filename_out_prefix> <n_graphs> <up_scale> <expand>\n";
        return 1;
    }

    std::string filename_in = argv[1];
    std::string filename_out = argv[2];
    uint32_t n_graphs = (uint32_t)std::stoul(argv[3]);
    uint32_t up_scale = (uint32_t)std::stoul(argv[4]);
    bool expand = (std::stoi(argv[5]) != 0);
    (void)expand; // not used

    if (n_graphs == 0) {
        std::cerr << "Error: n_graphs must be positive.\n";
        return 1;
    }

    std::ifstream infile(filename_in);
    if (!infile.is_open()) {
        std::cerr << "Error: cannot open input file " << filename_in << "\n";
        return 1;
    }

    uint32_t k_in = 0, n_rounds = 0, size_seed = 0;
    {
        std::string line;
        if (!std::getline(infile, line)) {
            std::cerr << "Error: empty input file.\n";
            return 1;
        }
        std::istringstream iss(line);
        if (!(iss >> k_in >> n_rounds >> size_seed)) {
            std::cerr << "Error: first line must be: k  n_rounds  size_seed\n";
            return 1;
        }
    }

    if (size_seed != 2) {
        std::cerr << "Error: this optimized KR generator assumes binary seed (size_seed=2).\n";
        return 1;
    }

    uint32_t k = k_in + up_scale;
    if (k > 63) {
        std::cerr << "Error: k=" << k << " too large for uint64 bitmask nodes (require k<=63).\n";
        return 1;
    }

    // Read alpha by number-of-zeros for the ORIGINAL k_in (k_in+1 lines), then extend to k+1.
    std::vector<double> alpha_by_zeros;
    alpha_by_zeros.reserve(k_in + 1);

    for (uint32_t i = 0; i < k_in + 1; ++i) {
        std::string line;
        if (!std::getline(infile, line)) {
            std::cerr << "Error: expected " << (k_in + 1) << " alpha lines after header.\n";
            return 1;
        }
        std::istringstream iss(line);
        double a;
        if (!(iss >> a)) {
            std::cerr << "Error: invalid alpha line at index " << i << ".\n";
            return 1;
        }
        if (a < 0.0 || a > 1.0) {
            std::cerr << "Error: alpha must be in [0,1]. Got " << a << ".\n";
            return 1;
        }
        alpha_by_zeros.push_back(a);
    }

    // Extend alpha if up_scale > 0: use average (matches your old logic).
    if (up_scale > 0) {
        double avg = 0.0;
        for (double x : alpha_by_zeros) avg += x;
        avg /= (double)alpha_by_zeros.size();
        for (uint32_t i = 0; i < up_scale; ++i) alpha_by_zeros.push_back(avg); // now size k+1
    }

    if (alpha_by_zeros.size() != (size_t)k + 1) {
        std::cerr << "Error: internal alpha size mismatch. Expected " << (k + 1)
                  << " got " << alpha_by_zeros.size() << ".\n";
        return 1;
    }

    // Read seed matrix 2x2
    double seed00=0, seed01=0, seed10=0, seed11=0;
    {
        std::string line;
        if (!std::getline(infile, line)) { std::cerr << "Error: missing seed row 0.\n"; return 1; }
        std::istringstream iss(line);
        if (!(iss >> seed00 >> seed01)) { std::cerr << "Error: invalid seed row 0.\n"; return 1; }
    }
    {
        std::string line;
        if (!std::getline(infile, line)) { std::cerr << "Error: missing seed row 1.\n"; return 1; }
        std::istringstream iss(line);
        if (!(iss >> seed10 >> seed11)) { std::cerr << "Error: invalid seed row 1.\n"; return 1; }
    }
    infile.close();

    // Validate seed is symmetric and values are probabilities
    auto in01 = [](double x){ return x >= 0.0 && x <= 1.0; };
    if (!in01(seed00) || !in01(seed01) || !in01(seed10) || !in01(seed11)) {
        std::cerr << "Error: seed entries must be in [0,1].\n";
        return 1;
    }
    if (std::abs(seed01 - seed10) > 1e-12) {
        std::cerr << "Error: seed must be symmetric for merged-mismatch optimization "
                     "(seed01 must equal seed10).\n";
        return 1;
    }

    // Convert alpha by zeros -> g_w by weight:
    // zeros z = k - w => g_w[w] = alpha_by_zeros[z]
    std::vector<double> g_w(k + 1, 0.0);
    for (uint32_t w = 0; w <= k; ++w) {
        uint32_t z = k - w;
        g_w[w] = alpha_by_zeros[z];
    }

    uint64_t n_nodes = 1ULL << k;
    std::cerr << "k=" << k << " (input " << k_in << " + up_scale " << up_scale << ")\n";
    std::cerr << "n_nodes=" << n_nodes << " (binary KR)\n";
    std::cerr << "n_rounds=" << n_rounds << "\n";

    // Precompute threshold table once
    KRPrecompSym T = build_precomp_symmetric_seed(k, n_rounds, seed00, seed01, seed11, g_w);

    // Generate graphs
    for (uint32_t gi = 0; gi < n_graphs; ++gi) {
        std::vector<FullClique> full_cliques;
        std::vector<FullBiclique> full_bicliques;
        std::vector<uint64_t> partial_edges;

        auto t_start = std::chrono::steady_clock::now();

        generate_edges_kr_pb_symmetric(
            n_rounds, k, n_nodes,
            g_w, T,
            full_cliques, full_bicliques, partial_edges
        );

        auto t_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t_end - t_start).count();
        std::cerr << "Graph " << gi << " generated in " << std::fixed << std::setprecision(3)
                  << sec << " s\n";

        // Write output
        std::ofstream out(filename_out + "_" + std::to_string(gi) + ".txt");
        if (!out.is_open()) {
            std::cerr << "Error: cannot open output file.\n";
            return 1;
        }

        out << "FULL_CLIQUES " << full_cliques.size() << "\n";
        for (auto &clq : full_cliques) {
            out << clq.w << " " << clq.nodes.size();
            for (uint64_t u : clq.nodes) out << " " << u;
            out << "\n";
        }

        out << "FULL_BICLIQUES " << full_bicliques.size() << "\n";
        for (auto &bcq : full_bicliques) {
            out << bcq.w_left << " " << bcq.w_right << " "
                << bcq.left.size() << " " << bcq.right.size();
            for (uint64_t u : bcq.left) out << " " << u;
            for (uint64_t v : bcq.right) out << " " << v;
            out << "\n";
        }

        out << "PARTIAL_EDGES " << partial_edges.size() << "\n";
        for (uint64_t e : partial_edges) {
            auto uv = decode_pair_u64(e, n_nodes);
            out << uv.first << " " << uv.second << "\n";
        }

        out.close();
    }

    return 0;
}