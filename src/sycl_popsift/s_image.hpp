#pragma once

#include "sycl/event.hpp" // not sure if I need to include the event specifically when including sycl

#include <sycl/sycl.hpp>

namespace popsift {

struct Image
{
    Image() = delete;

    Image(sycl::queue& q);

    /** Create a device-sided buffer of the given dimensions */
    Image(int w, int h, sycl::queue& q);

    ~Image();

    // void resetDimensions(int w, int h);
    void resetDimensions(int w, int h, int init_w, int init_h);

    sycl::event load(void* input);
    sycl::event load_divide(unsigned char* input);
    sycl::event load_divide_point(unsigned char* input, const int& scaled_w);

    inline int getWidth() const { return _w; }
    inline int getHeight() const { return _h; }

    // Simple function for testing and development
    sycl::event host_move(void* outptu);
    void print_region(int start_x, int start_y, int end_x, int end_y);

    inline float* getInput() { return _device_img; }

  protected:
    int _w;      // width  of current image
    int _h;      // height of current image
    int _max_w;  // allocated width  of image
    int _max_h;  // allocated height of image
    int _init_w; // Host image widht -- could be differnet due to apply scale factor done early
    int _init_h; // host image height

    sycl::queue& _device_queue; // reference to the queue

    // TODO: Test sycl image that is said to utlizie texture memory when availabe
    // also look into codeplays extension of sycl with bindless images and see if
    // it can make it more performant
    // unsigned char* _device_img;
    float* _device_img;
};
} // namespace popsift
