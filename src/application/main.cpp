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
static bool write_features = true;

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

          ("input-file,i", value<std::string>(&inputFile)->required(), "Input file");
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

    cerr << "Number of feature points: " << feature_list->getFeatureCount()
         << " number of feature descriptors: " << feature_list->getDescriptorCount() << endl;

    if(write_features)
    {
        fprintf(stderr, "\n\n\tWE ARE WRITING YEAHHH!\n");
        std::ofstream of("output-features.txt");
        feature_list->print(of, write_as_uchar);
    }

    delete feature_list;
}

int main(int argc, char** argv)
{
    popsift::Config config; // Init with default parameters
    list<string> inputFiles;
    string inputFile{};

    // cout << "Le upscalefactor: " << config.getUpscaleFactor() << endl;
    // cout << "Config gauus mode: " << config.getGaussMode()
    //      << "is same as: " << popsift::Config::VLFeat_Relative << endl;

    try
    {
        parseargs(argc,
                  argv,
                  &config,
                  inputFile); // Parse command line -- should add config
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
        cout << "current file: " << currFile << endl;
        SiftJob* job = process_image(currFile, PopSift);
        jobs.push(job);
    }

    // PopSift.allMainThread();

    while(!jobs.empty())
    {
        SiftJob* job = jobs.front();
        jobs.pop();
        if(job)
        {
            read_job(job);

            delete job;
        }
    }

    // unsigned char* image_data = nullptr;
    // int w, h;

    // TODO(jorgejen): Not really supposed to pass the image data to popsift in
    // final implenentation need to refactor!!! On sunday

    // Queue for multiple jobs
    // std::queue<SiftJob*> jobs;
    // SiftJob* leJob = process_image(inputFile, PopSift);

    // print job and wait for promise to be fufilled
    // int val = leJob->getHost();

    // std::cout << "The value resturned from future/promise: " << val <<
    // std::endl;

    // Testing the PopSift class

    // PopSift.printDim();
    // PopSift.printImageRegion(sycl::range(20, 25), sycl::range(20, 25));
    // PopSift.modifyImage();
    // PopSift.printImageRegion(sycl::range(20, 25), sycl::range(20, 25));
    // PopSift.printImage(10);
    // PopSift.printDevice();

    // PopSift.uninit(); // don't see the need as it will be done automatically
    // bu the destructor of the popsift class

    // fprintf(stderr, "\n\t\tI sleep now before return :D\n\n");
    fprintf(stderr, "\n\t\tExiting main now BYE\n\n");
    return EXIT_SUCCESS;
}
