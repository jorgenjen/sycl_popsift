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
static void parseargs(int argc, char** argv, popsift::Config* config, std::string& inputFile, std::string& cudaFile)
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
          "cuda-descs,c", value<std::string>(&cudaFile)->required(), "output file for metadata");
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

// std::vector<popsift::Descriptor> read_descriptors_from_file(const std::string& filename)
// {
//     std::ifstream in(filename, std::ios::binary | std::ios::ate);
//     if(!in)
//         throw std::runtime_error("Failed to open file");
//
//     std::streamsize size = in.tellg();
//     in.seekg(0, std::ios::beg);
//
//     size_t count = size / sizeof(popsift::Descriptor);
//     std::vector<popsift::Descriptor> data(count);
//
//     in.read(reinterpret_cast<char*>(data.data()), size);
//     in.close();
//
//     return data;
// }

std::vector<popsift::Feature> read_features_from_file(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if(!in)
        throw std::runtime_error("Failed to open file: " + filename);

    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    size_t count = size / sizeof(popsift::Feature);
    std::vector<popsift::Feature> data(count);

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

double xy_distance2(const popsift::Feature& a, const popsift::Feature& b)
{
    double dx = a.xpos - b.xpos;
    double dy = a.ypos - b.ypos;
    return dx * dx + dy * dy;
}

int find_xy_match(const popsift::Feature& f, const std::vector<popsift::Feature>& pool, const std::vector<bool>& used)
{
    double best = 1e30;
    int best_i = -1;

    for(int i = 0; i < pool.size(); i++)
    {
        if(used[i])
            continue;

        double d = xy_distance2(f, pool[i]);
        if(d < best)
        {
            best = d;
            best_i = i;
        }
    }

    const double THRESHOLD2 = 0.2; // 1 pixel squared
    if(best > THRESHOLD2)
    {
        printf("No match \n");
        return -1; // too far → no match
    }

    return best_i;
}

double l2_desc_distance(const popsift::Descriptor* a, const popsift::Descriptor* b)
{
    double sum = 0.0;
    for(int i = 0; i < 128; i++)
    {
        double diff = a->features[i] - b->features[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double cosine_similarity(const popsift::Descriptor* a, const popsift::Descriptor* b)
{
    double dot = 0.0, na = 0.0, nb = 0.0;

    for(int i = 0; i < 128; i++)
    {
        double x = a->features[i];
        double y = b->features[i];

        dot += x * y;
        na += x * x;
        nb += y * y;
    }

    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom == 0.0 ? 0.0 : dot / denom;
}

void compare_by_xy(const std::vector<popsift::Feature>& sycl, const std::vector<popsift::Feature>& cuda)
{
    std::vector<bool> used(cuda.size(), false);

    int matched = 0;
    double total_l2 = 0, max_l2 = 0;
    double total_cos = 0, min_cos = 1.0;

    for(const auto& f : sycl)
    {
        int idx = find_xy_match(f, cuda, used);
        if(idx < 0)
            continue;

        used[idx] = true;

        // double d = l2_desc_distance(f, cuda[idx]);
        // double c = cosine_similarity(f, cuda[idx]);

        double d = l2_desc_distance(f.desc[0], cuda[idx].desc[0]);
        double c = cosine_similarity(f.desc[0], cuda[idx].desc[0]);

        total_l2 += d;
        total_cos += c;

        if(d > max_l2)
            max_l2 = d;
        if(c < min_cos)
            min_cos = c;

        matched++;
    }

    std::cout << "Matched features: " << matched << "\n";
    std::cout << "Mean L2 descriptor distance: " << (total_l2 / matched) << "\n";
    std::cout << "Max L2 descriptor distance:  " << max_l2 << "\n";
    std::cout << "Mean cosine similarity: " << (total_cos / matched) << "\n";
    std::cout << "Min cosine similarity:  " << min_cos << "\n";
}

int find_best_orientation_match(const popsift::Feature& sycl_f, const popsift::Feature& cuda_f, int sycl_ori_idx)
{
    double best = 1e300;
    int best_idx = -1;

    double ori = sycl_f.orientation[sycl_ori_idx];

    for(int o = 0; o < cuda_f.num_ori; o++)
    {
        double diff = std::fabs(ori - cuda_f.orientation[o]);

        // Proper circular angle difference (wrap at 2π)
        diff = std::fmod(diff + M_PI, 2 * M_PI) - M_PI;
        diff = std::fabs(diff);

        if(diff < best)
        {
            best = diff;
            best_idx = o;
        }
    }

    return best_idx;
}
struct OrientationStats
{
    int comparisons = 0;
    double total_ori_err = 0.0;
    double max_ori_err = 0.0;

    double total_l2 = 0.0;
    double max_l2 = 0.0;

    double total_cos = 0.0;
    double min_cos = 1.0;
};

OrientationStats compare_orientations(const popsift::Feature& f_sycl, const popsift::Feature& f_cuda)
{
    OrientationStats stats;

    // for(int o = 0; o < f_sycl.num_ori; o++)
    // {

    int num_common = std::min(f_sycl.num_ori, f_cuda.num_ori);

    for(int o = 0; o < num_common; o++)
    {
        int m = find_best_orientation_match(f_sycl, f_cuda, o);
        if(m < 0)
            continue;

        // Orientation error
        double ori_err = std::fabs(f_sycl.orientation[o] - f_cuda.orientation[m]);
        ori_err = std::fmod(ori_err + M_PI, 2 * M_PI) - M_PI;
        ori_err = std::fabs(ori_err);

        stats.total_ori_err += ori_err;
        stats.max_ori_err = std::max(stats.max_ori_err, ori_err);

        // Descriptor comparison
        double l2 = l2_desc_distance(f_sycl.desc[o], f_cuda.desc[m]);
        double cos = cosine_similarity(f_sycl.desc[o], f_cuda.desc[m]);

        stats.total_l2 += l2;
        stats.max_l2 = std::max(stats.max_l2, l2);

        stats.total_cos += cos;
        stats.min_cos = std::min(stats.min_cos, cos);

        stats.comparisons++;
    }

    return stats;
}

void compare_all_orientations(const std::vector<popsift::Feature>& sycl, const std::vector<popsift::Feature>& cuda)
{
    std::vector<bool> used(cuda.size(), false);

    int matched_keypoints = 0;
    OrientationStats global;

    for(const auto& f : sycl)
    {
        int idx = find_xy_match(f, cuda, used);
        if(idx < 0)
            continue;

        if(f.num_ori != cuda[idx].num_ori)
        {
            printf("Ori diff %d\n", f.num_ori - cuda[idx].num_ori);
        }

        used[idx] = true;
        matched_keypoints++;

        OrientationStats s = compare_orientations(f, cuda[idx]);

        // Accumulate global stats
        global.comparisons += s.comparisons;
        global.total_ori_err += s.total_ori_err;
        global.max_ori_err = std::max(global.max_ori_err, s.max_ori_err);
        global.total_l2 += s.total_l2;
        global.max_l2 = std::max(global.max_l2, s.max_l2);
        global.total_cos += s.total_cos;
        global.min_cos = std::min(global.min_cos, s.min_cos);
    }

    std::cout << "Matched keypoints: " << matched_keypoints << "\n";
    std::cout << "Orientation comparisons: " << global.comparisons << "\n";

    std::cout << "Mean orientation error (rad): " << (global.total_ori_err / global.comparisons) << "\n";
    std::cout << "Max orientation error (rad):  " << global.max_ori_err << "\n";

    std::cout << "Mean L2 descriptor distance:  " << (global.total_l2 / global.comparisons) << "\n";
    std::cout << "Max L2 descriptor distance:   " << global.max_l2 << "\n";

    std::cout << "Mean cosine similarity:       " << (global.total_cos / global.comparisons) << "\n";
    std::cout << "Min cosine similarity:        " << global.min_cos << "\n";
}

// std::vector<popsift::Feature> read_features(const std::string& filename)
// {
//     std::ifstream in(filename, std::ios::binary);
//     if(!in)
//     {
//         throw std::runtime_error("Failed to open file for reading: " + filename);
//     }
//
//     uint64_t count;
//     in.read(reinterpret_cast<char*>(&count), sizeof(count));
//
//     std::vector<popsift::Feature> feats(count);
//
//     for(size_t i = 0; i < count; i++)
//     {
//         popsift::Feature& f = feats[i];
//
//         in.read(reinterpret_cast<char*>(&f.xpos), sizeof(float));
//         in.read(reinterpret_cast<char*>(&f.ypos), sizeof(float));
//         in.read(reinterpret_cast<char*>(&f.sigma), sizeof(float));
//         in.read(reinterpret_cast<char*>(&f.num_ori), sizeof(int));
//
//         for(int o = 0; o < f.num_ori; o++)
//         {
//             in.read(reinterpret_cast<char*>(&f.orientation[o]), sizeof(float));
//
//             // Allocate a descriptor
//             f.desc[o] = new popsift::Descriptor();
//
//             // Read the 128 descriptor floats
//             in.read(reinterpret_cast<char*>(f.desc[o]->features), sizeof(float) * 128);
//         }
//     }
//
//     return feats;
// }

std::vector<popsift::Feature> read_features(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if(!in)
        throw std::runtime_error("Failed to open file");

    std::vector<popsift::Feature> feats;

    while(true)
    {
        popsift::Feature f;

        if(!in.read((char*)&f.xpos, sizeof(float)))
            break; // EOF
        in.read((char*)&f.ypos, sizeof(float));
        in.read((char*)&f.sigma, sizeof(float));
        in.read((char*)&f.num_ori, sizeof(int));

        for(int o = 0; o < f.num_ori; o++)
        {
            in.read((char*)&f.orientation[o], sizeof(float));

            f.desc[o] = new popsift::Descriptor();
            in.read((char*)f.desc[o]->features, sizeof(float) * 128);
        }

        feats.push_back(f);
    }

    return feats;
}

int main(int argc, char** argv)
{
    popsift::Config config; // Init with default parameters
    string inputFile{};
    string cudaFile{};

    try
    {
        parseargs(argc, argv, &config, inputFile, cudaFile);
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

    // std::vector<popsift::Descriptor> desc_pool = read_descriptors_from_file(inputFile);
    // std::vector<popsift::Descriptor> desc_pool_cuda = read_descriptors_from_file(cudaFile);

    // std::vector<popsift::Feature> desc_pool_cuda = read_features_from_file(cudaFile);
    // std::vector<popsift::Feature> desc_pool = read_features_from_file(inputFile);

    std::vector<popsift::Feature> desc_pool = read_features(inputFile);
    std::vector<popsift::Feature> desc_pool_cuda = read_features(cudaFile);

    int desc_pool_size = desc_pool.size();

    std::printf("sycl size %zu \ncuda size %zu", desc_pool.size(), desc_pool_cuda.size());

    // compare_by_xy(desc_pool, desc_pool_cuda);
    compare_all_orientations(desc_pool, desc_pool_cuda);

    //

    // popsift::FeaturesDev sycl_dev(dev_q, desc_pool.size(), desc_pool);
    // popsift::FeaturesDev cuda_dev(dev_q, desc_pool_cuda.size(), desc_pool_cuda);

    // popsift::FeaturesDev sycl_dev(dev_q, 10, desc_pool);
    // popsift::FeaturesDev cuda_dev(dev_q, 10, desc_pool_cuda);
    //
    // std::vector<std::string> metadata;
    // metadata.push_back("SSD1,SSD2,match");
    //
    // auto [match_matrix, matrix_wait, matrix_free] = sycl_dev.matchAndReturn(&cuda_dev);

    // matrix_wait();

    // int count = 0;
    // for(int i = 0; i < sycl_dev.getDescriptorCount(); i++)
    // {
    //     sycl::vec<int, 3>& match = match_matrix[i];
    //     if(match.z())
    //     {
    //         // const popsift::Feature* l_f = sycl_dev->getFeatureForDescriptor(i);
    //         // const popsift::Feature* r_f = rFeatures->getFeatureForDescriptor(match.x());
    //         // cout << setprecision(5) << showpoint << "point (" << l_f->xpos << "," << l_f->ypos << ") in l
    //         matches
    //         "
    //         //      << "point (" << r_f->xpos << "," << r_f->ypos << ") in r -- " << "i = " << i
    //         //      << " matc.x() = " << match.x() << endl;
    //         count++;
    //     }
    //
    //     // if(match.x() != 0)
    //     // {
    //     //     fprintf(stderr, "Match matrix %d, %d, %d \n", match.x(), match.y(), match.z());
    //     // }
    // }
    // cout << "Match count normal: " << count << endl << endl;

    // matrix_free();

    // popsift::FeaturesDev l_dev(dev_q, l_index_set.size(), build_descriptor(l_index_set, desc_pool));
    // popsift::FeaturesDev r_dev(dev_q, r_index_set.size(), build_descriptor(r_index_set, desc_pool));

    // std::ofstream of();
    // of << metadata[0] << std::endl;
    // for(int i = 1; i < metadata.size(); ++i)
    // {
    //     of << metadata[i] << std::endl;
    // }

    return EXIT_SUCCESS;
}
