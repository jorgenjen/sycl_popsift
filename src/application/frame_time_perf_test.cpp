#include "sycl_popsift/features.hpp"

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
static void parseargs(int argc, char** argv, popsift::Config* config, std::string& warmupFiles, std::string& testFiles)
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

          ("test-file,i", value<std::string>(&testFiles)->required(), "Test file")(
            "warmup-file,w", value<std::string>(&warmupFiles)->required(), "Warmup file");
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
    // cout << "Loading " << w << " x " << h << " image " << inputFile << endl;

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

    // fprintf(stderr,
    //         "\nNumbmer of features points: %d  number of feature descriptors: %d\n",
    //         feature_list->getFeatureCount(),
    //         feature_list->getDescriptorCount());

    if(write_features)
    {
        std::ofstream of("output-features.txt");
        feature_list->print(of, write_as_uchar);
    }

    delete feature_list;
}

std::mutex mtx;
std::condition_variable job_sender;
// std::condition_variable job_reciever;

void schedule_jobs(popsift::SyncQueue<SiftJob*>& jobs,
                   int& frames_in_flight,
                   list<string>& warmupFiles,
                   list<string>& inputFiles,
                   PopSift& PopSift)
{
#define MAX_FRAMES_IN_FLIGHT 20

#define DO_WARMUP 1
#if DO_WARMUP
    for(const auto& currFile : warmupFiles)
    {
        std::unique_lock<std::mutex> lock(mtx);
        // frames_in_flight.wait(0, std::memory_order_relaxed, [](int val) { return val >= 5; });
        job_sender.wait(lock, [&] { return frames_in_flight < MAX_FRAMES_IN_FLIGHT; });
        frames_in_flight++;
        lock.unlock();
        SiftJob* job = process_image(currFile, PopSift);
        jobs.push(job);
    }

#endif

    // The actuall test

#define TEST_ITERATIONS 20
    for(int i = 0; i < TEST_ITERATIONS; ++i)
    {
        printf("Doing iteration %i\n", i);
        for(const auto& currFile : inputFiles)
        {
            // printf("Scheduling %s\n", currFile.c_str());
            std::unique_lock<std::mutex> lock(mtx);
            // frames_in_flight.wait(0, std::memory_order_relaxed, [](int val) { return val >= 5; });
            job_sender.wait(lock, [&] { return frames_in_flight < MAX_FRAMES_IN_FLIGHT; });
            frames_in_flight++;
            lock.unlock();
            SiftJob* job = process_image(currFile, PopSift);
            jobs.push(job);
        }
    }

    // fprintf(stderr, "\n\tDone sheduling \n");
    jobs.push(nullptr); // To signal that we are done
}

void retrive_jobs(popsift::SyncQueue<SiftJob*>& jobs,
                  int warmup_count,
                  int& frames_in_flight,
                  std::vector<std::string>& imgNames)
{
    std::vector<std::string> metadata;
    metadata.push_back("filename,feature_count,descriptor_count");
    int count = 0;
    // while(!jobs.empty())

    SiftJob* job;
    while((job = jobs.pull()) != nullptr)
    {
        // SiftJob* job = jobs.front();
        // jobs.pop();
        if(job)
        {
            popsift::FeaturesHost* feature_list = job->getHost(); // wait for job to complete

            if(count >= warmup_count)
            {
                // write -- Actual test data (not warmup)
                std::ostringstream oss;

                oss << feature_list->getFeatureCount() << "," << feature_list->getDescriptorCount();
                metadata.push_back(oss.str());
                // fprintf(stderr,
                //         "\nNumbmer of features points: %d  number of feature descriptors: %d\n",
                //         feature_list->getFeatureCount(),
                //         feature_list->getDescriptorCount());
            }
            // else
            // {
            //     fprintf(stderr, "Water fuck\n");
            // }

            delete feature_list;
            delete job;
            {
                std::lock_guard<std::mutex> lock(mtx);
                frames_in_flight--;
            }
            job_sender.notify_one();
        }
        // else
        // {
        //     fprintf(stderr, "Not a job somehow... \n");
        // }

        count++; // Just to ensure we don't write for warmup data
        // fprintf(stderr, "We roling around! \n");
    }

    // Write results
    std::ofstream of("test_metadata.csv");
    of << metadata[0] << std::endl;
    for(int i = 1; i < metadata.size(); ++i)
    {
        of << imgNames[(i - 1) % imgNames.size()] << "," << metadata[i] << std::endl;
    }
}

int main(int argc, char** argv)
{
    popsift::Config config; // Init with default parameters

    list<string> warmupFiles; // For avoiding cold start and cuda module loading
    string warmupFile{};

    list<string> inputFiles;
    string inputFile{};

    // cout << "Le upscalefactor: " << config.getUpscaleFactor() << endl;
    // cout << "Config gauus mode: " << config.getGaussMode()
    //      << "is same as: " << popsift::Config::VLFeat_Relative << endl;

    try
    {
        parseargs(argc, argv, &config, warmupFile, inputFile); // Parse command line -- should add config
                                                               // as parameter so it can be modified
        std::cout << inputFile << std::endl;
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Used for warmup
    if(boost::filesystem::exists(warmupFile))
    {
        if(boost::filesystem::is_directory(warmupFile))
        {
            collectFilenames(&warmupFiles, warmupFile);
            if(warmupFiles.empty())
            {
                cerr << "Empty warmup directory... Exiting" << endl;
                return EXIT_SUCCESS;
            }
        }
        else if(boost::filesystem::is_regular_file(warmupFile))
        {
            std::cout << "Just one warmup image" << std::endl;
            warmupFiles.push_back(warmupFile);
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

    // Files used in the test
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

    std::vector<std::string> imgNames;

    for(const auto& currFile : inputFiles)
    {
        size_t pos = currFile.find_last_of('/');

        std::string fileName = (pos != std::string::npos) ? currFile.substr(pos + 1) : currFile;

        // Storing file name to vector for indexing
        imgNames.push_back(fileName);
    }

    PopSift PopSift(config);

    // std::queue<SiftJob*> jobs;
    popsift::SyncQueue<SiftJob*> jobs;

    int frames_in_flight = 0;

    // std::thread sender(schedule_jobs(jobs, frames_in_flight, warmupFiles, inputFiles, PopSift));
    // std::thread sender([&] { schedule_jobs(jobs, frames_in_flight, warmupFiles, inputFiles, PopSift); });
    //
    // printf("warmupFiles count = %zu\n", warmupFiles.size());
    // std::thread receiver([&] { retrive_jobs(jobs, warmupFiles.size(), frames_in_flight, imgNames); });

    std::thread sender(schedule_jobs,
                       std::ref(jobs),
                       std::ref(frames_in_flight),
                       std::ref(warmupFiles),
                       std::ref(inputFiles),
                       std::ref(PopSift));

    // Start retrive_jobs in another thread
    std::thread receiver(retrive_jobs,
                         std::ref(jobs),
                         static_cast<int>(warmupFiles.size()),
                         std::ref(frames_in_flight),
                         std::ref(imgNames));

    sender.join();
    receiver.join();

    // #define DO_WARMUP 1
    // #if DO_WARMUP
    //     for(const auto& currFile : warmupFiles)
    //     {
    //         // cout << "current file: " << currFile << endl;
    //         SiftJob* job = process_image(currFile, PopSift);
    //         jobs.push(job);
    //     }
    //
    //     while(!jobs.empty())
    //     {
    //         SiftJob* job = jobs.front();
    //         jobs.pop();
    //         if(job)
    //         {
    //             popsift::FeaturesHost* feature_list = job->getHost(); // wait for job to complete
    //
    //             // Don't care about this data -- Just for warmup
    //             delete feature_list;
    //
    //             delete job;
    //         }
    //     }
    // #endif
    //
    // // Conduct actual test
    // #define TEST_ITERATIONS 2
    //     for(int i = 0; i < TEST_ITERATIONS; ++i)
    //     {
    //         for(const auto& currFile : inputFiles)
    //         {
    //             // cout << "current file: " << currFile << endl;
    //             SiftJob* job = process_image(currFile, PopSift);
    //             jobs.push(job);
    //         }
    //     }
    //
    //     while(!jobs.empty())
    //     {
    //         SiftJob* job = jobs.front();
    //         jobs.pop();
    //         if(job)
    //         {
    //             popsift::FeaturesHost* feature_list = job->getHost(); // wait for job to complete
    //
    //             std::ostringstream oss;
    //
    //             oss << feature_list->getFeatureCount() << "," << feature_list->getDescriptorCount();
    //             metadata.push_back(oss.str());
    //             fprintf(stderr,
    //                     "\nNumbmer of features points: %d  number of feature descriptors: %d\n",
    //                     feature_list->getFeatureCount(),
    //                     feature_list->getDescriptorCount());
    //
    //             delete feature_list;
    //             delete job;
    //         }
    //     }
    //
    //     std::ofstream of("test_metadata.csv");
    //     of << metadata[0] << std::endl;
    //     for(int i = 1; i < metadata.size(); ++i)
    //     {
    //         of << imgNames[i % imgNames.size()] << "," << metadata[i] << std::endl;
    //     }
    // Write file name and feature and desc count so that we have that to correlate with the nsight systems data
    // collected

    return EXIT_SUCCESS;
}
