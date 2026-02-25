// main.cpp – Traveling Salesman Problem (TSP) Solver using a Parallel Genetic Algorithm
//
// Overview:
//   Reads a list of cities with (x, y) coordinates from a text file, builds a
//   pairwise distance matrix in parallel, then evolves a population of candidate
//   tours using a genetic algorithm until a maximum generation count is reached.
//   The best tour found and its total length are written to an output file.
//   Per-generation logging is printed to stdout so convergence can be monitored.
//
// Genetic-algorithm operators:
//   Selection  – Tournament selection: the shorter of two randomly chosen tours wins.
//   Crossover  – Order Crossover (OX1): copies a random segment from parent A, then
//                fills the remaining positions in the order they appear in parent B,
//                guaranteeing a valid (no duplicate city) permutation.
//   Mutation   – Swap mutation: with a small probability, two random cities in the
//                tour are swapped.
//   Elitism    – The single best tour of each generation is always carried forward,
//                preventing loss of the current best solution.
//
// Parallelism (mutex-free):
//   Distance matrix – rows are divided among threads; each thread writes only its
//                     own rows, so no mutex or atomic is needed.
//   Evolution step  – the new generation is split into equal-sized chunks that
//                     collectively cover exactly the whole population (no empty
//                     slice), each processed by a dedicated thread with its own
//                     independent RNG seeded from the master generator.
//
// Input file format:
//   One city per line in CSV format: <id>,<x>,<y>,,,,,, (trailing commas/fields
//   are ignored).  The first field is a numeric city ID and is discarded.
//
// Build:
//   g++ -std=c++17 -O2 -pthread -o tsp main.cpp
//
// Usage:
//   ./tsp [input_file] [output_file] [pop_size] [max_gen] [mut_rate]
//   Defaults: cities.txt  result.txt  200  500  0.02

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ─────────────────────────────── data structures ─────────────────────────────

struct City {
    double x{};
    double y{};
};

// A Tour is a permutation of city indices in [0, numCities).
using Tour = std::vector<int>;

// ──────────────────────────────── helper functions ───────────────────────────

// Returns the Euclidean distance between two cities.
static inline double euclideanDist(const City& a, const City& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Returns the total length of a tour (including the return edge to the start).
static double tourLength(const Tour& tour,
                         const std::vector<std::vector<double>>& dist) {
    double total = 0.0;
    const int n = static_cast<int>(tour.size());
    for (int i = 0; i < n; ++i)
        total += dist[tour[i]][tour[(i + 1) % n]];
    return total;
}

// ─────────────────────── parallel distance matrix ────────────────────────────

// Worker function: fills rows [rowStart, rowEnd) of the distance matrix.
// Each thread receives an exclusive, non-overlapping row range, so no mutex
// is required for concurrent writes.
static void computeDistRows(const std::vector<City>& cities,
                            std::vector<std::vector<double>>& distMatrix,
                            int rowStart,
                            int rowEnd) {
    const int n = static_cast<int>(cities.size());
    for (int i = rowStart; i < rowEnd; ++i)
        for (int j = 0; j < n; ++j)
            distMatrix[i][j] = euclideanDist(cities[i], cities[j]);
}

// Builds and returns the full n×n distance matrix using as many threads as
// there are hardware cores (capped at n so no thread gets an empty range).
static std::vector<std::vector<double>>
buildDistanceMatrix(const std::vector<City>& cities) {
    const int n = static_cast<int>(cities.size());
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));

    // Never spawn more threads than there are rows to process.
    const int hwThreads = static_cast<int>(std::thread::hardware_concurrency());
    const int numThreads = std::max(1, std::min(hwThreads, n));

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    // Distribute rows as evenly as possible; earlier threads get one extra row
    // when the row count is not evenly divisible.
    const int chunkSize = n / numThreads;
    const int remainder = n % numThreads;
    int rowStart = 0;
    for (int t = 0; t < numThreads; ++t) {
        const int rowEnd = rowStart + chunkSize + (t < remainder ? 1 : 0);
        threads.emplace_back(computeDistRows,
                             std::cref(cities),
                             std::ref(dist),
                             rowStart,
                             rowEnd);
        rowStart = rowEnd;
    }
    for (auto& thr : threads)
        thr.join();

    return dist;
}

// ──────────────────────────── genetic-algorithm operators ────────────────────

// Creates an initial population of random tours.
static std::vector<Tour>
initPopulation(int popSize, int numCities, std::mt19937& rng) {
    Tour base(numCities);
    std::iota(base.begin(), base.end(), 0);  // 0, 1, 2, …, numCities-1

    std::vector<Tour> pop(popSize, base);
    for (auto& tour : pop)
        std::shuffle(tour.begin(), tour.end(), rng);
    return pop;
}

// Tournament selection: returns a reference to the better (shorter) of two
// randomly chosen individuals in the current population.
static const Tour&
tournamentSelect(const std::vector<Tour>& pop,
                 const std::vector<double>& fitness,
                 std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0,
                                            static_cast<int>(pop.size()) - 1);
    const int a = pick(rng);
    const int b = pick(rng);
    return fitness[a] < fitness[b] ? pop[a] : pop[b];
}

// Order Crossover (OX1): copies a random contiguous segment from parentA, then
// fills the remaining positions in the order the cities appear in parentB.
// The result is always a valid permutation (every city appears exactly once).
static Tour orderCrossover(const Tour& parentA,
                           const Tour& parentB,
                           std::mt19937& rng) {
    const int n = static_cast<int>(parentA.size());
    std::uniform_int_distribution<int> pick(0, n - 1);
    int lo = pick(rng), hi = pick(rng);
    if (lo > hi) std::swap(lo, hi);

    // Start with an empty child; -1 marks unfilled positions.
    Tour child(n, -1);

    // Copy the segment [lo, hi] from parentA.
    for (int i = lo; i <= hi; ++i)
        child[i] = parentA[i];

    // Fill remaining positions in the order cities appear in parentB,
    // skipping any city already present in the child.
    int pos = (hi + 1) % n;
    for (int k = 0; k < n; ++k) {
        const int city = parentB[(hi + 1 + k) % n];
        if (std::find(child.begin(), child.end(), city) == child.end()) {
            child[pos] = city;
            pos = (pos + 1) % n;
        }
    }
    return child;
}

// Swap mutation: with probability mutationRate, swaps two randomly chosen
// cities in the tour in place.
static void swapMutate(Tour& tour, double mutationRate, std::mt19937& rng) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    if (prob(rng) < mutationRate) {
        std::uniform_int_distribution<int> pick(0,
                                                static_cast<int>(tour.size()) - 1);
        std::swap(tour[pick(rng)], tour[pick(rng)]);
    }
}

// ────────────────────────── parallel evolution step ──────────────────────────

// Worker function: fills nextPop[lo..hi) with offspring produced by
// tournament selection, OX1 crossover, and swap mutation.
// Each thread receives its own seeded RNG so there is no shared mutable state.
static void evolveChunk(const std::vector<Tour>& currentPop,
                        const std::vector<double>& fitness,
                        std::vector<Tour>& nextPop,
                        double mutationRate,
                        int lo,
                        int hi,
                        unsigned seed) {
    std::mt19937 rng(seed);
    for (int i = lo; i < hi; ++i) {
        const Tour& p1 = tournamentSelect(currentPop, fitness, rng);
        const Tour& p2 = tournamentSelect(currentPop, fitness, rng);
        nextPop[i] = orderCrossover(p1, p2, rng);
        swapMutate(nextPop[i], mutationRate, rng);
    }
}

// ─────────────────────────────── file I/O ────────────────────────────────────

// Reads city coordinates from a CSV file where each line has the format:
//   <id>,<x>,<y>,,,,,, (any trailing comma-separated fields are ignored).
// Throws std::runtime_error if the file cannot be opened or contains no data.
static std::vector<City> readCities(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open())
        throw std::runtime_error("Cannot open input file: " + path);

    std::vector<City> cities;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        // Parse: id,x,y,...
        std::istringstream ss(line);
        std::string token;
        // Skip id field
        if (!std::getline(ss, token, ',')) continue;
        // Read x
        std::string xStr, yStr;
        if (!std::getline(ss, xStr, ',')) continue;
        if (!std::getline(ss, yStr, ',')) continue;
        City c;
        try {
            c.x = std::stod(xStr);
            c.y = std::stod(yStr);
        } catch (...) {
            continue;
        }
        cities.push_back(c);
    }

    if (cities.empty())
        throw std::runtime_error("No valid city data found in file: " + path);
    return cities;
}

// Writes the best tour and its total length to a text file.
// Throws std::runtime_error if the file cannot be opened.
static void writeTour(const std::string& path,
                      const Tour& tour,
                      double length) {
    std::ofstream fout(path);
    if (!fout.is_open())
        throw std::runtime_error("Cannot open output file: " + path);

    fout << std::fixed << std::setprecision(4);
    fout << "Best tour length: " << length << "\n";
    fout << "Tour order (0-indexed city indices):\n";
    for (const int city : tour)
        fout << city << "\n";
}

// ────────────────────────────────── main ─────────────────────────────────────

int main(int argc, char* argv[]) {
    // Default parameters – all can be overridden via the command line.
    std::string inputFile  = "cities.txt";
    std::string outputFile = "result.txt";
    int         popSize    = 200;
    int         maxGen     = 500;
    double      mutRate    = 0.02;

    if (argc >= 2) inputFile  = argv[1];
    if (argc >= 3) outputFile = argv[2];
    if (argc >= 4) popSize    = std::atoi(argv[3]);
    if (argc >= 5) maxGen     = std::atoi(argv[4]);
    if (argc >= 6) mutRate    = std::atof(argv[5]);

    // ── load cities ──────────────────────────────────────────────────────────
    std::vector<City> cities;
    try {
        cities = readCities(inputFile);
    } catch (const std::runtime_error& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    const int numCities = static_cast<int>(cities.size());
    std::cout << "Loaded " << numCities << " cities.\n";

    // ── build distance matrix in parallel (mutex-free) ───────────────────────
    const auto distMatrix = buildDistanceMatrix(cities);

    // ── initialise population ────────────────────────────────────────────────
    std::mt19937 masterRng(std::random_device{}());
    auto population = initPopulation(popSize, numCities, masterRng);

    // ── evolution loop ───────────────────────────────────────────────────────
    // Use at most as many threads as individuals so no thread gets an empty slice.
    const int hwThreads  = static_cast<int>(std::thread::hardware_concurrency());
    const int numThreads = std::max(1, std::min(hwThreads, popSize));

    Tour   bestTour;
    double bestLength = std::numeric_limits<double>::max();

    std::cout << std::left
              << std::setw(10) << "Gen"
              << std::setw(20) << "Best length"
              << "Best tour (first 10 cities)\n"
              << std::string(60, '-') << "\n";

    for (int gen = 0; gen < maxGen; ++gen) {
        // Evaluate every tour's total length (fitness = shorter is better).
        std::vector<double> fitness(popSize);
        for (int i = 0; i < popSize; ++i)
            fitness[i] = tourLength(population[i], distMatrix);

        // Track the global best individual.
        const int eliteIdx = static_cast<int>(
            std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
        if (fitness[eliteIdx] < bestLength) {
            bestLength = fitness[eliteIdx];
            bestTour   = population[eliteIdx];
        }

        // Log progress at the first generation, every 50 thereafter, and at the end.
        if (gen == 0 || gen % 50 == 0 || gen == maxGen - 1) {
            std::cout << std::left
                      << std::setw(10) << gen
                      << std::fixed << std::setprecision(2)
                      << std::setw(20) << bestLength;
            // Print up to the first 10 city indices of the best tour.
            const int preview = std::min(10, numCities);
            for (int k = 0; k < preview; ++k)
                std::cout << bestTour[k] << (k + 1 < preview ? "-" : "");
            if (numCities > 10) std::cout << "-...";
            std::cout << "\n";
        }

        // Build the next generation in parallel.
        // Chunk boundaries are computed so that together they cover [0, popSize)
        // exactly – no thread receives an empty range.
        std::vector<Tour> nextPop(popSize);
        std::vector<std::thread> threads;
        threads.reserve(numThreads);

        const int chunkSize = popSize / numThreads;
        const int remainder = popSize % numThreads;
        int lo = 0;
        for (int t = 0; t < numThreads; ++t) {
            const int hi = lo + chunkSize + (t < remainder ? 1 : 0);
            threads.emplace_back(evolveChunk,
                                 std::cref(population),
                                 std::cref(fitness),
                                 std::ref(nextPop),
                                 mutRate,
                                 lo,
                                 hi,
                                 masterRng() ^ static_cast<unsigned>(t));
            lo = hi;
        }
        for (auto& thr : threads)
            thr.join();

        // Elitism: overwrite the worst individual in the new generation with
        // the best individual from the current generation.
        const int worstIdx = static_cast<int>(
            std::max_element(fitness.begin(), fitness.end()) - fitness.begin());
        nextPop[worstIdx] = bestTour;

        population = std::move(nextPop);
    }

    // ── print final summary ──────────────────────────────────────────────────
    std::cout << "\nEvolution complete.\n";
    std::cout << "Best tour length: " << std::fixed << std::setprecision(4)
              << bestLength << "\n";

    // ── write result to file ─────────────────────────────────────────────────
    try {
        writeTour(outputFile, bestTour, bestLength);
        std::cout << "Result written to " << outputFile << "\n";
    } catch (const std::runtime_error& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
