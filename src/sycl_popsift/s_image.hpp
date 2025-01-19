#pragma once

#include <sycl/sycl.hpp>

namespace popsift {

struct Image 
{
  Image() = delete;

  Image(sycl::queue& q);

  /** Create a device-sided buffer of the given dimensions */
  Image( int w, int h, sycl::queue& q);

  ~Image( );


  void resetDimensions( int w, int h );

  void load( void* input );

protected:
  int _w;     // width  of current image
  int _h;     // height of current image
  int _max_w; // allocated width  of image
  int _max_h; // allocated height of image

  sycl::queue& _device_queue; // reference to the queue

  unsigned char* _device_img;

};
}
