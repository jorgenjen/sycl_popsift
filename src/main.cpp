#include <cstdio>
#include <iostream>
#include <sycl/sycl.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#ifdef USE_DEVIL
/* #include <devil_cpp_wrapper.hpp> */
#include <IL/il.h>
#include <IL/ilu.h>
#endif






/* static void parseargs(int argc, char** argv, std::string& inputFile) { */
/*   // should switch to the boost library later on -- as in popsift */
/*   if (argc < 2) { */
/*     throw std::runtime_error("Usage: " + std::string(argv[0]) + " <input file>"); */
/*   } */
/*   inputFile = std::string(argv[1]); */
/* } */


// should probably use a similar options struct as popsift in the future revisions
// just for initial layout
static void parseargs(int argc, char** argv, std::string& inputFile) {
    using namespace boost::program_options;

    options_description options("Options");
    {
        options.add_options()
            ("help,h", "Print usage")
            /* ("verbose,v", bool_switch()->notifier([&](bool i) {if(i) config.setVerbose(); }), "") */
            /* ("log,l", bool_switch()->notifier([&](bool i) {if(i) config.setLogMode(popsift::Config::All); }), "Write debugging files") */

            ("input-file,i", value<std::string>(&inputFile)->required(), "Input file");
    
    }
    options_description all("Allowed options");
    
    // currently just options
    /* all.add(options).add(parameters).add(modes).add(informational); */
    all.add(options);
    variables_map vm;
    
    try
    {    
       store(parse_command_line(argc, argv, all), vm);

       if (vm.count("help")) {
           std::cout << all << '\n';
           exit(EXIT_SUCCESS);
       }

        notify(vm); // Notify does processing (e.g., raise exceptions if required args are missing)
    }
    catch(boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        std::cerr << "Usage:\n\n" << all << std::endl;
        exit(EXIT_FAILURE);
    }
}




// void processImageOld(const std::string& inputFile, unsigned char& image_data)
// {
//   using namespace std;
//
//   // load in the image 
//
// #ifdef USE_DEVIL
//
//   /* unsigned char* image_data; */
//
//   cout << "USING DEVIL" << endl;
//
//   ilImage img;
//   if( img.Load( inputFile.c_str() ) == false ) {
//     cerr << "Could not load image " << inputFile << endl;
//     return 0;
//   }
//   if( img.Convert( IL_LUMINANCE ) == false ) {
//     cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << endl;
//     exit( -1 );
//   }
//   const auto w = img.Width();
//   const auto h = img.Height();
//   cout << "Loading " << w << " x " << h << " image " << inputFile << endl;
//
//   // stores the image to given pointer
//   image_data = img.GetData();
//
//   // should probably use a version of this class
//   /* job = PopSift.enqueue( w, h, image_data ); */
//
//
//   img.Clear();
//
//
// #else
//   cout << "Devil not enabled, cannot load image backup not implemented yet :D" << endl;
// #endif
// }

// image_data is a reference to a pointer so that we can update the nullptr to the image data from devIL
void processImage(const std::string& inputFile, unsigned char* &image_data, int &w, int &h)
{
  using namespace std;
  // load in the image 

#ifdef USE_DEVIL
    // Initialize DevIL
    ilInit();

    // Generate and bind an image handle
    ILuint image;
    ilGenImages(1, &image);
    ilBindImage(image);

    // Load the image
    if (!ilLoadImage(inputFile.c_str())) {
        cerr << "Could not load image " << inputFile << endl;
        ilDeleteImages(1, &image); // Clean up
        // return -1;
    }

    // Convert to grayscale (luminance)
    if (!ilConvertImage(IL_LUMINANCE, IL_UNSIGNED_BYTE)) {
        cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << endl;
        ilDeleteImages(1, &image); // Clean up
        // return -1;
    }

    // Get image dimensions
    w = ilGetInteger(IL_IMAGE_WIDTH);
    h = ilGetInteger(IL_IMAGE_HEIGHT);
    cout << "Loading " << w << " x " << h << " image " << inputFile << endl;

    // Get raw image data
    image_data = ilGetData();


  // print out first 10 bytes of the image
  // for (int i = 0; i < 10; i++) {
  //   std::cout << static_cast<int>(image_data[i*10]) << " ";
  // }

    // Example usage of image_data with your PopSift class
    // job = PopSift.enqueue(w, h, image_data);

    // Clean up the DevIL image
    // ilDeleteImages(1, &image);
    // need to clean it up later on 

#else
  cout << "Devil not enabled, cannot load image backup not implemented yet :D" << endl;
#endif
}


// const std::string secret{
//   "Ifmmp-!xpsme\"\012J(n!tpssz-!Ebwf/!"
//   "J(n!bgsbje!J!dbo(u!ep!uibu/!.!IBM\01"};
//
// const auto sz = secret.size();


int main(int argc, char **argv)
{
  {
    using namespace std;
    #ifdef USE_DEVIL
      cout << "Devil is enabled" << endl;
    #else
      cout << "Devil is not enabled" << endl;
    #endif
  }

  std::string         inputFile{};

  try {
    parseargs( argc, argv, inputFile ); // Parse command line
    std::cout << inputFile << std::endl;
  }
  catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }


  // check the image
  if( boost::filesystem::exists( inputFile ) ) {
    if( boost::filesystem::is_directory( inputFile ) ) {
      std::cout << "BOOST " << inputFile << " is directory -- Multiple files are currently not supported" << std::endl;
      // Will support multiple files later on
      /* collectFilenames( inputFiles, inputFile ); */
      /* if( inputFiles.empty() ) { */
      /*     cerr << "No files in directory, nothing to do" << endl; */
      /*     return EXIT_SUCCESS; */
      /* } */
    } else if( boost::filesystem::is_regular_file( inputFile ) ) {

      std::cout << "Regurlar file will be processed" << std::endl;
      /* inputFiles.push_back( inputFile ); */
    } else {
      std::cout << "Input file is neither regular file nor directory, nothing to do" << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    std::cout << "Input file does not exist, nothing to do" << std::endl;
  }



  unsigned char* image_data = nullptr;
  int w, h;
  processImage(inputFile, image_data, w, h);
  

  // unsigned char test_img[] = {0, 1, 9, 255, 255, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4};

  // unsigned char* test_p_img = &test_img[0];
  // print out the first 10 bytes of the image
  for (int i = 0; i < 10; i++) {
    std::cout << static_cast<int>(image_data[i]) << " ";
  }
  std::cout << std::endl;


  std::cout << "Image size: " << w << " x " << h << std::endl;


  {
    using namespace sycl;

    try {
      queue q;


      std::cout << "Selected device: "
        << q.get_device().get_info<info::device::name>()
        << "\n";


      // seems like sycl::image does only support four dimensional images hence seems to not make alot of sense for a grayscale image
      // hence using buffers instead. Might look at codeplay bindless images later on and see if that would work well for grayscale


      buffer<unsigned char, 2> srcImage(image_data, range<2>(w, h));

      // the same as above just different syntax
      // auto srcImage = buffer<unsigned char, 2>(image_data, range<2>(w, h));

      // even more verbose -- same as above
      // buffer<unsigned char, 2> srcImage = buffer<unsigned char, 2>(image_data, range<2>(w, h));


      // unsigned char* result = malloc_shared<unsigned char>(w*h, q);


      // unsigned char* res = std::malloc(w*h);
      unsigned char* res = static_cast<unsigned char*>(std::malloc(w * h * sizeof(unsigned char)));


      buffer<unsigned char, 2> dstImage(res, range<2>(w, h));


      q.submit([&](handler& cgh) {

        accessor img(srcImage, cgh, read_only);
        accessor result(dstImage, cgh, write_only);
        cgh.parallel_for(range<2>(w, h), [=](id<2> idx) {
          result[idx] = img[idx] - 1;
        });
      });

      q.wait();

      // print out the first 10 bytes of the image

      for (int i = 0; i < 10; i++) {
        std::cout << static_cast<int>(res[i]) << " ";
      }
      std::cout << std::endl;

    } catch (const sycl::exception& e) {
      std::cout << "Exception caught: " << e.what() << std::endl;
    }






      // auto img = srcImage.get_access<access::mode::read>(cgh);

    // char* result = malloc_shared<char>(sz, q);
    // std::memcpy(result, secret.data(), sz);
    //
    // q.parallel_for(sz, [=](auto& i) {
    //   result[i] -= 1;
    // }).wait();
    //
    // std::cout << result << "\n";
    // free(result, q);
    return 0;


  }


}


