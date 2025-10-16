#include "sycl_popsift/sift_extremum.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Popsift includes
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <sycl_popsift/features.hpp>
#include <sycl_popsift/non_sycl/sift_conf.hpp>
#include <sycl_popsift/popsift.hpp>
#include <sycl_popsift/sift_extremum.h>
#include <unistd.h>

#include <chrono> // only for test
#include <list>
#include <thread> // For testing

#ifdef USE_DEVIL
#include <IL/il.h>
#include <IL/ilu.h>
#endif

using namespace std;

// should probably use a similar options struct as popsift in the future
// revisions just for initial layout
static void parseargs(int argc, char** argv, popsift::Config* config, std::string& inputFile, std::string& outputFile)
{
    // using namespace boost::program_options;
    using boost::program_options::options_description;
    using boost::program_options::parse_command_line;
    using boost::program_options::value;
    using boost::program_options::variables_map;

    options_description options("Options");
    {
        options.add_options()("help,h",
                              "Print usage")("input-file,i", value<std::string>(&inputFile)->required(), "Input file")(
          "output-file,o", value<std::string>(&outputFile)->required(), "output file for metadata");
    }
    options_description all("Allowed options");

    all.add(options);
    variables_map vm;

    try
    {
        store(parse_command_line(argc, argv, all), vm);

        if(vm.count("help"))
        {
            std::cout << all << '\n';
            exit(EXIT_SUCCESS);
        }

        notify(vm); // Notify does processing (e.g., raise exceptions if
                    // required args are missing)
    }
    catch(boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        std::cerr << "Usage:\n\n" << all << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::vector<popsift::Descriptor> read_descriptors_from_file(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if(!in)
        throw std::runtime_error("Failed to open file");

    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    size_t count = size / sizeof(popsift::Descriptor);
    std::vector<popsift::Descriptor> data(count);

    in.read(reinterpret_cast<char*>(data.data()), size);
    in.close();

    return data;
}

std::vector<popsift::Descriptor> build_descriptor(std::unordered_set<int> indecies,
                                                  const std::vector<popsift::Descriptor>& desc_pool)
{
    std::vector<popsift::Descriptor> desc;
    desc.reserve(indecies.size());
    for(auto i : indecies)
    {
        desc.emplace_back(desc_pool[i]);
    }
    return desc;
}

int main(int argc, char** argv)
{
    popsift::Config config; // Init with default parameters
    string inputFile{};
    string outputFile{};

    try
    {
        parseargs(argc, argv, &config, inputFile, outputFile);
        std::cout << inputFile << std::endl;
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if(boost::filesystem::exists(inputFile))
    {
        std::cout << "File exist will load from: " << inputFile << std::endl;
    }
    else
    {
        std::cout << "Input file does not exist, nothing to do" << std::endl << "Exiting..." << std::endl;
        return EXIT_FAILURE;
    }

    // PopSift PopSift(config);
    sycl::queue dev_q;

    std::vector<popsift::Descriptor> desc_pool = read_descriptors_from_file(inputFile);
    int desc_pool_size = desc_pool.size();

    if(desc_pool_size <= 4000000) // Want to compare 200k with 200k and have them be unique
    {
        printf("Trying to generate two disticnt sets from a pool with less elements that the toatal of the two sets\n "
               "Exiting...");
        return EXIT_FAILURE;
    }
    printf("Despool size = %d", desc_pool_size);

    int num = 1000; // Need to be outer loop variant

    std::mt19937 rng(48); // For repetability
    std::uniform_int_distribution<> dist(0, desc_pool_size - 1);

    std::vector<std::string> metadata;
    metadata.push_back("l_desc_size,r_desc_size,matches");

#define WARMUP_RUNS 5

#define TEST_JOINT_MATRIX true
#define MAX_DESC_SIZE 5000 // Was 50000
#define STEP 1000

    bool doing_warmup = false;
    int warmup_count = 0;
    for(int i = STEP; i <= MAX_DESC_SIZE; i += STEP)
    {
        for(int j = STEP; j <= MAX_DESC_SIZE; j += STEP)
        {
            printf("(%d, %d) -- iter %d/%d\n",
                   i,
                   j,
                   (i / STEP) * (MAX_DESC_SIZE / STEP) + (j / STEP),
                   MAX_DESC_SIZE / STEP * MAX_DESC_SIZE / STEP);
            if(warmup_count < WARMUP_RUNS)
            {
                warmup_count++;
                i = 2000;
                j = 2000;
            }
            if(warmup_count == WARMUP_RUNS)
            {
                // start the actual test warmup is done
                warmup_count++;
                i = STEP;
                j = STEP;
                printf("NOW WE START FOR REAL\n\n\n");
            }

            std::unordered_set<int> l_index_set;
            std::unordered_set<int> r_index_set;

            while(l_index_set.size() < i)
            {
                int a = dist(rng);
                l_index_set.insert(a); // Only inserts if not already present
            }

            while(r_index_set.size() < j)
            {
                int a = dist(rng);
                if(l_index_set.count(a) != 0)
                    continue;
                r_index_set.insert(a); // Only insert if unique to both sets
            }

            popsift::FeaturesDev l_dev(dev_q, l_index_set.size(), build_descriptor(l_index_set, desc_pool));
            popsift::FeaturesDev r_dev(dev_q, r_index_set.size(), build_descriptor(r_index_set, desc_pool));

#if TEST_JOINT_MATRIX
            l_dev.compute_squared_norms();
            r_dev.compute_squared_norms();

            // fprintf(stderr, "Computing norm is guuchi!\n");
            auto [match_matrix, matrix_wait, matrix_free] = l_dev.preNormMatrixMatchAndReturn(&r_dev);
            // fprintf(stderr, " Now we have started the matrix compute");
#else
            auto [match_matrix, matrix_wait, matrix_free] = l_dev.matchAndReturn(&r_dev);
#endif
            matrix_wait();

            int count = 0;
            for(int idx = 0; idx < l_dev.getDescriptorCount(); ++idx)
            {
                sycl::vec<int, 3>& match = match_matrix[idx];
                if(match.z())
                {
                    count++;
                }
            }
            std::ostringstream oss;
            oss << i << "," << j << "," << count;
            metadata.push_back(oss.str());

            matrix_free();
        }
    }

    std::ofstream of(outputFile);
    of << metadata[0] << std::endl;
    for(int i = 1; i < metadata.size(); ++i)
    {
        of << metadata[i] << std::endl;
    }

    return EXIT_SUCCESS;
}
