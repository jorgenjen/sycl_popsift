#include "popsift.hpp"
#include "sycl/accessor.hpp"

#include <sycl/sycl.hpp>
#include <iostream>

using namespace std;

PopSift::PopSift(int w, int h, unsigned char* imageData)
  : _w(w),
  _h(h),
  // _imageData(imageData, sycl::range<2>(w, h), {sycl::property::buffer::use_host_ptr()})
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


// Helper function for development
// ranges are inclusive on 0th dimension and exclusive on 1th dimension
void PopSift::printImageRegion(sycl::range<2> horiz, sycl::range<2> vert)
{
  // print out the first 10 bytes of the image
  using namespace sycl;

  // wait for all previous enqued task to end before doing the print to show desired data
  _deviceQueue.wait();

  host_accessor<unsigned char, 2, access::mode::read> h_acc(_imageData);

  if (vert.get(0) > _w && vert.get(0) < 0 || 
      vert.get(1) > _w && vert.get(1) < 0 ||
      vert.get(0) >= vert.get(1)
  )
  {
    std::cout << "Image region is not legal" << std::endl;
  }

  std::cout << "Image region: horiz = (" << horiz.get(0) << " -> " << horiz.get(1)
            << ") vert = (" << vert.get(0) << " -> " << vert.get(1) << ")" << std::endl;
  // using range in a odd way (I know :D)
  for (int i = vert.get(0); i < vert.get(1); ++i)
  {
    for (int j = horiz.get(0); j < horiz.get(1); ++j)
    {
         std::printf("%03u ", h_acc[j][i]);
    }
       std::cout << std::endl;
  }
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
    printf("w=%d  -- h=%d", _w, _h);

    accessor img(_imageData, cgh, read_write);
    cgh.parallel_for(range<2>(_w, _h), [=](id<2> idx) {
      img[idx] = img[idx] - 1;
    });
  });

}
