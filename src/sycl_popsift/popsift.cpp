#include "popsift.hpp"

#include <sycl/sycl.hpp>
#include <iostream>

using namespace std;

PopSift::PopSift(int w, int h, unsigned char* imageData)
  : _w(w),
  _h(h),
  _imageData(imageData, sycl::range<2>(w, h))
{

  // sycl::queue _deviceQueue;
  // sycl::buffer<unsigned char, 2> _imageData(imageData, sycl::range<2>(_w, _h));

  cout << "PopSift constructor" << endl;

}

void PopSift::printDim()
{
  cout << "Width: " << _w << endl;
}

void PopSift::printDevice()
{
  {
    using namespace sycl;

    try {
      queue q;

      std::cout << "Selected device in PopSift method using SYCL: "
        << q.get_device().get_info<info::device::name>()
        << "\n";
    } catch (const sycl::exception& e) {
      std::cout << "Exception caught: " << e.what() << std::endl;
    }

  }
}

void PopSift::printImage()
{
  // print out the first 10 bytes of the image

  using namespace sycl;


  // MR segfault
  _deviceQueue.submit([&](handler& cgh) {

    accessor img(_imageData, cgh, read_only);
    
    // int printCount = std::min(static_cast<size_t>(10), _w *_h);
    int printCount = 10;

    for (size_t i = 0; i < printCount; ++i)
    {
      size_t row = i / _w;
      size_t col = i % _w;
      unsigned char pixel_val = img[row][col];

      std::cout << static_cast<int>(pixel_val); 
    }
    std::cout << std::endl;

    // for (int i = 0; i < 10; i++) {
    //   // std::cout << static_cast<int>(img[i]) << " ";
    //   id<2> idx(i, 0);
    //   std::cout << img[idx] << " ";
    // }
    // std::cout << std::endl;
  });
}


void PopSift::modifyImage()
{
  using namespace sycl;
  try {

    std::cout << "Selected device in PopSift method (modifyImage) using SYCL: "
      << _deviceQueue.get_device().get_info<info::device::name>()
      << "\n";
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Modify the image
  
  std::cout << "Modifyig image now" << std::endl;

  _deviceQueue.submit([&](handler& cgh) {

    accessor img(_imageData, cgh, read_write);
    cgh.parallel_for(range<2>(_w, _h), [=](id<2> idx) {
      img[idx] = img[idx] - 1;
    });
  });

}
