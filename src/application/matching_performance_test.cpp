
// #include "sycl_popsift/features.hpp"
#include "sycl_popsift/common/assist.h" // For initQueue;
#include "sycl_popsift/features.hpp"
#include "sycl_popsift/sift_extremum.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <sycl/sycl.hpp>
#include <sycl_popsift/sift_desc_config.hpp> // For FeatureType

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <random> // For random number generation for slice of pool
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Popsift includes
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <sycl_popsift/non_sycl/sift_conf.hpp>
#include <sycl_popsift/popsift.hpp>
#include <unistd.h>

#include <chrono> // only for test
#include <list>
#include <thread> // For testing

#ifdef USE_DEVIL
#include <IL/il.h>
#include <IL/ilu.h>
#endif

using namespace std;

static bool write_as_uchar = false;
static bool write_features = false; // Takse loads of time when running many

// should probably use a similar options struct as popsift in the future
// revisions just for initial layout
static void parseargs(int argc, char** argv, popsift::Config* config, std::string& inputFile)
{
    // using namespace boost::program_options;
    using boost::program_options::options_description;
    using boost::program_options::parse_command_line;
    using boost::program_options::value;
    using boost::program_options::variables_map;

    options_description options("Options");
    {
        options.add_options()("help,h", "Print usage")
          /* ("verbose,v", bool_switch()->notifier([&](bool i) {if(i)
             config.setVerbose(); }), "") */
          /* ("log,l", bool_switch()->notifier([&](bool i) {if(i)
             config.setLogMode(popsift::Config::All); }), "Write debugging
             files")
           */

          ("input-file,i",
           value<std::string>(&inputFile)->required(),
           "Used for descriptor pool to samle from in test");
    }
    // options_description modes("Modes");
    // {
    //     modes.add_options()("cpu-only",
    //                         boost::program_options::bool_switch()->notifier([&](bool a) {
    //                             if(a)
    //                                 config->setCpuOnly();
    //                         }),
    //                         "Use CPU only no accelerators.");
    // }

    options_description all("Allowed options");

    // currently just options
    /* all.add(options).add(parameters).add(modes).add(informational); */
    // all.add(options).add(modes);
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

static void collectFilenames(list<string>* inputFiles, const boost::filesystem::path& inputFile)
{
    std::vector<boost::filesystem::path> vec;
    std::copy(boost::filesystem::directory_iterator(inputFile),
              boost::filesystem::directory_iterator(),
              std::back_inserter(vec));
    for(const auto& currPath : vec)
    {
        if(boost::filesystem::is_regular_file(currPath))
        {
            inputFiles->push_back(currPath.string());
        }
        else if(boost::filesystem::is_directory(currPath))
        {
            collectFilenames(inputFiles, currPath);
        }
    }
}

// image_data is a reference to a pointer so that we can update the nullptr to
// the image data from devIL
SiftJob* process_image(const std::string& inputFile, PopSift& PopSift)
{
    SiftJob* job;
    unsigned char* image_data;
    int w, h;
    // unsigned char* image_data; // should move image_data to local varaible

#ifdef USE_DEVIL
    // Initialize DevIL
    ilInit();

    // Generate and bind an image handle
    ILuint image;
    ilGenImages(1, &image);
    ilBindImage(image);

    // Load the image
    if(!ilLoadImage(inputFile.c_str()))
    {
        cerr << "Could not load image " << inputFile << endl;
        ilDeleteImages(1, &image); // Clean up
        return nullptr;
    }

    // Convert to grayscale
    if(!ilConvertImage(IL_LUMINANCE, IL_UNSIGNED_BYTE))
    {
        cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << endl;
        ilDeleteImages(1, &image); // Clean up
                                   // return -1;
    }

    w = ilGetInteger(IL_IMAGE_WIDTH);
    h = ilGetInteger(IL_IMAGE_HEIGHT);
    cout << "Loading " << w << " x " << h << " image " << inputFile << endl;

    // Get raw image data
    image_data = ilGetData();

    // enqueue the job - image is copied in this method
    job = PopSift.enqueue(w, h, image_data);

    // Clean up the DevIL image -- can't do it here need to be after we are done
    // with it
    ilDeleteImages(1, &image);
    // need to clean it up later on

    return job;

#else
    cout << "Devil not enabled, cannot load image backup not implemented yet :D" << endl;
#endif
}

void read_job(SiftJob* job)
{
    popsift::FeaturesHost* feature_list = job->getHost(); // wait for job to complete

    // cerr << "\nNumber of feature points: " << feature_list->getFeatureCount()
    //      << " number of feature descriptors: " << feature_list->getDescriptorCount() << endl;

    fprintf(stderr,
            "\nNumbmer of features points: %d  number of feature descriptors: %d\n",
            feature_list->getFeatureCount(),
            feature_list->getDescriptorCount());

    if(write_features)
    {
        std::ofstream of("output-features.txt");
        feature_list->print(of, write_as_uchar);
    }

    delete feature_list;
}

bool uniqueVal(std::vector<int> idx_sequence, int idx)
{
    // printf("size of existing sequence = %zu", idx_sequence.size());
    for(int i = 0; i < static_cast<int>(idx_sequence.size()); ++i)
    {
        if(idx_sequence[i] == idx)
            return false;
    }
    return true;
}

popsift::Descriptor* generateRandDescSequenceArray(int size,
                                                   std::vector<std::array<FeatureType, 128>>& desc_pool,
                                                   mt19937& gen,
                                                   uniform_int_distribution<>& distrib)
{
    if(size > desc_pool.size())
    {
        std::cerr << "Need to use one descriptor multiple times in sequence due to sequence lenght being " << size
                  << " which is more than the number of descriptors in pool(" << desc_pool.size()
                  << ") \n\tHence we are exiting now without writing results.  Add more images to directory to have a "
                     "large enough descriptor pool!... \n Exiting..."
                  << std::endl;
        exit(1);
    }

    // mt19937 gen(seed);
    // uniform_int_distribution<> distrib(0, desc_pool.size());

    // struct Descriptor
    // {
    //     FeatureType features[128];
    // };
    popsift::Descriptor* ptr = (popsift::Descriptor*)malloc(size * sizeof(popsift::Descriptor));

    std::vector<int> idx_sequence;
    for(int i = 0; i < size; ++i)
    {
        // printf("IDX seq size = %d\n", static_cast<int>(idx_sequence.size()));
        int idx;
        do
        {
            idx = distrib(gen);
        } while(!uniqueVal(idx_sequence, idx));

        idx_sequence.push_back(idx); // Stored to ensure no duplicate descriptors used

        memcpy(ptr[i].features, &(desc_pool[idx]), sizeof(popsift::Descriptor));
    }

    // Store generated sequence to ptr

    // free(ptr);
    return ptr;
}

struct matrixMatchBenchInfo
{
    double l_norm_start;
    double l_orm_end;

    double r_norm_start;
    double r_norm_end;

    double match_start;
    double match_end;
};

#if USE_JOINT_MATRIX
void benchmarkMarixMatchingPerformance(std::vector<std::array<FeatureType, 128>>& desc_pool, int seed, sycl::queue& Q)
{
    //
    popsift::FeaturesDev lFeatures(Q); // left
    popsift::FeaturesDev rFeatures(Q); // right

    int lSize = 8150;
    int rSize = 7890;

    lFeatures.reset(lSize, lSize); // Use same value we only care about descriptors in this case
    rFeatures.reset(rSize, rSize);

    mt19937 gen(seed);
    uniform_int_distribution<> distrib(0, desc_pool.size());

    // popsift::Descriptor* lSequenceArray = generateRandDescSequenceArray(20000, desc_pool, gen, distrib);
    // popsift::Descriptor* rSequenceArray = generateRandDescSequenceArray(20000, desc_pool, gen, distrib);

    popsift::Descriptor* lSequenceArray;
    popsift::Descriptor* rSequenceArray;

    // int lSize = 8150;
    // int rSize = 7890;
    for(int i = 0; i < rSize; ++i)
    {
        lSequenceArray = (popsift::Descriptor*)malloc(lSize * sizeof(popsift::Descriptor));

        memcpy(lSequenceArray[i].features, &(desc_pool[i]), sizeof(popsift::Descriptor));
    }

    for(int i = 0; i < rSize; ++i)
    {
        rSequenceArray = (popsift::Descriptor*)malloc(rSize * sizeof(popsift::Descriptor));

        // memcpy(desc_pool[i].features, &(desc_pool[idx]), sizeof(popsift::Descriptor));
        memcpy(rSequenceArray[i].features, &(desc_pool[i + lSize]), sizeof(popsift::Descriptor));
    }

    // features.getDescriptors();

    // Copy is not a part of the benchmark
    // sycl::event lTransfer = Q.memcpy(lFeatures.getDescriptors(), lSequenceArray, 512 * sizeof(popsift::Descriptor));
    // sycl::event rTransfer = Q.memcpy(rFeatures.getDescriptors(), rSequenceArray, 512 * sizeof(popsift::Descriptor));

    sycl::event lTransfer = Q.memcpy(lFeatures.getDescriptors(), lSequenceArray, lSize * sizeof(popsift::Descriptor));
    sycl::event rTransfer = Q.memcpy(rFeatures.getDescriptors(), rSequenceArray, rSize * sizeof(popsift::Descriptor));

    lTransfer.wait();
    rTransfer.wait();

    lFeatures.compute_squared_norms();
    rFeatures.compute_squared_norms();

    // auto [match_matrix, matrix_wait, matrix_free] = lFeatures.preNormMatrixMatchAndReturn(&(rFeatures));
    auto [match_matrix, matrix_wait, matrix_free] = lFeatures.matchAndReturn(&(rFeatures));

    matrix_wait();

    int count = 0;
    for(int i = 0; i < lFeatures.getDescriptorCount(); i++)
    {
        sycl::vec<int, 3>& match = match_matrix[i];
        if(match.z())
        {
            const popsift::Feature* l_f = lFeatures.getFeatureForDescriptor(i);
            const popsift::Feature* r_f = rFeatures.getFeatureForDescriptor(match.x());
            cout << setprecision(5) << showpoint << "point (" << l_f->xpos << "," << l_f->ypos << ") in l matches "
                 << "point (" << r_f->xpos << "," << r_f->ypos << ") in r -- " << "i = " << i
                 << " matc.x() = " << match.x() << endl;
            count++;
        }
    }
    cout << "Match matrix: " << count << endl << endl;

    matrixMatchBenchInfo info{};

    // double match_start = matrix_match_event;
    // double match_end = matrix_match_e matrix_remainder_event;

#if QUEUE_PROFILING
    sycl::event evt = lFeatures.getNormsEvent();

    // info
    //   .l_norm_start

    evt.wait();
    double frame_start = evt.template get_profiling_info<sycl::info::event_profiling::command_start>();

    double frame_end = evt.template get_profiling_info<sycl::info::event_profiling::command_end>();

    double frame_time = frame_end - frame_start;

    printf("Time to compute squared norms = %lf ns == %lf ms\n\n", frame_time, frame_time / 1000000);
#endif

    free(lSequenceArray);
    free(rSequenceArray);
}
#endif

int main(int argc, char** argv)
{
    popsift::Config config; // Init with default parameters
    list<string> inputFiles;
    string inputFile{};

    try
    {
        parseargs(argc, argv, &config, inputFile);
        // Parse command line -- should add config
        // as parameter so it can be modified
        std::cout << inputFile << std::endl;
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // check the image
    if(boost::filesystem::exists(inputFile))
    {
        if(boost::filesystem::is_directory(inputFile))
        {
            collectFilenames(&inputFiles, inputFile);
            if(inputFiles.empty())
            {
                cerr << "No files in directory, nothing to do" << endl;
                return EXIT_SUCCESS;
            }
        }
        else if(boost::filesystem::is_regular_file(inputFile))
        {
            std::cout << "Regurlar file will be processed" << std::endl;
            inputFiles.push_back(inputFile);
        }
        else
        {
            std::cout << "Input file is neither regular file nor directory, "
                         "nothing to do"
                      << std::endl
                      << "Exiting..." << std::endl;
            return EXIT_FAILURE;
        }
    }
    else
    {
        std::cout << "Input file does not exist, nothing to do" << std::endl << "Exiting..." << std::endl;
        return EXIT_FAILURE;
    }

    PopSift PopSift(config);

    std::queue<SiftJob*> jobs;
    for(const auto& currFile : inputFiles)
    {
        SiftJob* job = process_image(currFile, PopSift);
        jobs.push(job);
    }

    // std::vector<std::array<FeatureType, 128>> desc_pool; // Pool to select from
    // Vector for easy growth as performance does not matter here
    std::vector<popsift::FeaturesDev*> featuresArray;

    while(!jobs.empty())
    {
        SiftJob* job = jobs.front();
        jobs.pop();
        if(job)
        {
            featuresArray.push_back(job->getDev());
        }
    }

    // Loop over each and match with each

    for(int i = 0; i < featuresArray.size(); ++i)
    {
        for(int j = 0; j < featuresArray.size(); ++j)
        {
            if(i == j)
                continue; // skip matching with self

            auto [match_matrix, matrix_wait, matrix_free] =
              featuresArray[i]->preNormMatrixMatchAndReturn(featuresArray[j]);
            matrix_wait();

            matrix_free();
        }
    }

#if false

    // I need to make this work to get the good scatter polot but sampling data using 
    while(!jobs.empty())
    {
        SiftJob* job = jobs.front();
        jobs.pop();
        if(job)
        {
            popsift::FeaturesHost* feature_list = job->getHost(); // wait for job to complete

            fprintf(stderr,
                    "\nNumber of feature descriptors: %d --> Current pool total = %zu\n",
                    feature_list->getDescriptorCount(),
                    desc_pool.size() + feature_list->getDescriptorCount());

            popsift::Descriptor* descs = feature_list->getDescriptors();
            int num_descriptors = feature_list->getDescriptorCount();

            for(int i = 0; i < num_descriptors; ++i)
            {
                printf("descs[%d] --> %f\n", i, static_cast<float>(descs[i].features[0]));
                std::array<FeatureType, 128> descriptor;
                std::memcpy(descriptor.data(), descs[i].features, sizeof(FeatureType) * 128);
                desc_pool.push_back(descriptor);
            }

            delete feature_list;
            delete job;
        }
    }
    for(int i = 0; i < 99; ++i)
    {
        printf("desc_pool[%d][0] -> %f\n", i, static_cast<float>(desc_pool[i][0]));
    }

    // Now we have a full vecor of many descriptors that we can use to make our test of the matching we want to test
    // Make new memory segment (malloc) for each test so that we don't get any caching as that would not be real result
    // Ensure that left and right does not have any overlapping indecies as that would not happen much in reality that
    // you have two exactly same vecors which might give different performance as that would always win the lader
    // significatnly everytime( not sure if that affects much tbf)

    sycl::queue Q = popsift::initQueue();
    popsift::FeaturesDev features(Q);

    bool matrixSupport = popsift::supportsJointMatrixMatch(Q);

    if(matrixSupport)
    {
// Joint matrix matching benchmark
#if USE_JOINT_MATRIX
        benchmarkMarixMatchingPerformance(desc_pool, 42, Q);
#else
        fprintf(stderr,
                "Joint matrix matcing is supported but the code was configured with -DJointMatrix=OFF hence cannot "
                "bencmark it!\n");
#endif
    }
    else
    {
        std::cerr
          << "JointMatrix matching is not supported... This could be due to not cmake not setting -DJointMatrix=ON or "
             "it coudl be that your device: "
          << Q.get_device().get_info<sycl::info::device::name>()
          << " Does not support the JointMatrix extension or that your sycl environment/compiler does not support it"
          << std::endl;
    }
#endif

    return EXIT_SUCCESS;
}
