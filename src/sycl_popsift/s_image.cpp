#include "s_image.hpp"


#include <sycl/sycl.hpp>
#include <iostream>

namespace popsift 
{

Image::Image(sycl::queue& q) 
  : _w(0), _h(0)
  , _max_w(0), _max_h(0),
  _device_queue(q)
{
}


Image::Image( int w, int h, sycl::queue& q)
  : _w(w), _h(h)
  , _max_w(w), _max_h(h),
  _device_queue(q)
{
  // allocate( w, h );
  // need to allocate malloc_device
  _device_img = sycl::malloc_device<unsigned char>(w*h, q);
  if (_device_img == nullptr)
    std::cout << "Could not allocate segment -- failsafe not implemented so odd bahaviour could happen" << std::endl;
}

Image::~Image( )
{
    if( _max_w == 0 ) return;

    sycl::free(_device_img, _device_queue);
    // destroyTexture( );
    // _input_image_d.freeDev( );
    // _input_image_h.freeHost( popsift::CudaAllocated );
}


void Image::resetDimensions(int w, int h)
{
  if( _max_w == 0 && _max_h == 0 ) {
    // First time instantiating
    _max_w = _w = w;
    _max_h = _h = h;
    // allocate( w, h );
    _device_img = sycl::malloc_device<unsigned char>(w*h, _device_queue);
    if (_device_img == nullptr)
      std::cout << "Could not allocate segment -- failsafe not implemented so odd bahaviour could happen" << std::endl;
    return;
  }

  if( w == _w && h == _h ) return;
  /* everything OK, nothing to do */

  if( w*h <= _max_w*_max_h)
  {
    // smaller than current allocated segment hence it is fine to reuse
      // works aslong as we are using simle memory segment that is one dimensional like this one
    _w = w;
    _h = h;
    return;
  }

  // larger than current segment hence need to free and remalloc

  // TODO: See if sycl has realloc -- could not find it
  sycl::free(_device_img, _device_queue);

  _max_w = _w = w;
  _max_h = _h = h;
  _device_img = sycl::malloc_device<unsigned char>(w*h, _device_queue);
  if (_device_img == nullptr)
    std::cout << "Could not allocate segment -- failsafe not implemented so odd bahaviour could happen" << std::endl;

}

void Image::load( void* input )
{
  _device_queue.memcpy(_device_img, input, _w*_h);
}

}


