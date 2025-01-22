#include "s_image.hpp"

#include <sycl/sycl.hpp>

#include <iostream>

namespace popsift {

Image::Image(sycl::queue& q)
  : _w(0)
  , _h(0)
  , _max_w(0)
  , _max_h(0)
  , _device_queue(q)
{}

Image::Image(int w, int h, sycl::queue& q)
  : _w(w)
  , _h(h)
  , _max_w(w)
  , _max_h(h)
  , _device_queue(q)
{
    // allocate( w, h );
    // need to allocate malloc_device
    _device_img = sycl::malloc_device<unsigned char>(w * h, q);
    if(_device_img == nullptr)
        std::cout << "Could not allocate segment -- failsafe not implemented so odd bahaviour could happen"
                  << std::endl;
}

Image::~Image()
{
    if(_max_w == 0)
        return;

    sycl::free(_device_img, _device_queue);
    // destroyTexture( );
    // _input_image_d.freeDev( );
    // _input_image_h.freeHost( popsift::CudaAllocated );
}

void Image::resetDimensions(int w, int h)
{
    if(_max_w == 0 && _max_h == 0)
    {
        // First time instantiating
        _max_w = _w = w;
        _max_h = _h = h;
        // allocate( w, h );
        _device_img = sycl::malloc_device<unsigned char>(w * h, _device_queue);
        if(_device_img == nullptr)
            std::cout << "Could not allocate segment -- failsafe not implemented so odd bahaviour could happen"
                      << std::endl;
        return;
    }

    if(w == _w && h == _h)
        return;
    /* everything OK, nothing to do */

    if(w * h <= _max_w * _max_h)
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
    _device_img = sycl::malloc_device<unsigned char>(w * h, _device_queue);
    if(_device_img == nullptr)
        std::cout << "Could not allocate segment -- failsafe not implemented so odd bahaviour could happen"
                  << std::endl;
}

sycl::event Image::load(void* input) { return _device_queue.memcpy(_device_img, input, _w * _h); }

// only for printing and debugging
sycl::event Image::host_move(void* output) { return _device_queue.memcpy(output, _device_img, _w * _h); };

// only for printing and debugging -- quite inefficient
void Image::print_region(int start_x, int start_y, int end_x, int end_y)
{
    using std::cout;
    using std::endl;
    using std::printf;
    cout << "Inside of Print_region of Image" << endl << endl << endl;

    unsigned char* img = (unsigned char*)malloc(_w * _h * sizeof(unsigned char));
    if(img == NULL)
    {
        cout << "Memory allocation failed" << endl;
        return;
    }
    sycl::event write_event = host_move(img);

    if(start_x > _w || end_x > _w || start_y > _h || end_y > _h)
    {
        cout << "Region coordinates are outisde of bounds of Image" << endl;
        return;
    }
    if(start_x > end_x || start_y > end_y)
    {
        cout << "Invalid region" << endl;
        return;
    }
    if(start_x < 0 || start_y < 0 || end_x < 0 || end_y < 0)
    {
        cout << "Cannot have negative position of region" << endl;
        return;
    }
    printf("Image region (%d, %d) -> (%d, %d)\n", start_x, start_y, end_x, end_y);

    try
    {
        write_event.wait();
    }
    catch(const sycl::exception& e)
    {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return;
    }
    // _device_queue.wait(); // wait for memcpy to finish
    for(int i = start_y; i < end_y; ++i)
    {
        for(int j = start_x; j < end_x; ++j)
        {
            printf("%03u ", img[i * _w + j]);
        }
        cout << endl;
    }
    cout << endl << endl;
    free(img);
}

} // namespace popsift
