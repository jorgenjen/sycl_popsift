#include "popsift.hpp"

#include <sycl/sycl.hpp>
#include <iostream>

using namespace std;

PopSift::PopSift(int w, int h)
  : _w(w),
  _h(h) 
{
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
