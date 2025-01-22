#pragma once

#include "sycl/event.hpp"

#include <sycl/sycl.hpp>

namespace popsift {

struct Image
{
    Image() = delete;

    Image(sycl::queue& q);

    /** Create a device-sided buffer of the given dimensions */
    Image(int w, int h, sycl::queue& q);

    ~Image();

    void resetDimensions(int w, int h);

    sycl::event load(void* input);

    inline int getWidth() const { return _w; }
    inline int getHeight() const { return _h; }

    // Simple function for testing and development
    sycl::event host_move(void* outptu);
    void print_region(int start_x, int start_y, int end_x, int end_y);

  protected:
    int _w;     // width  of current image
    int _h;     // height of current image
    int _max_w; // allocated width  of image
    int _max_h; // allocated height of image

    sycl::queue& _device_queue; // reference to the queue

    // TODO: Test sycl image that is said to utlizie texture memory when availabe
    // also look into codeplays extension of sycl with bindless images and see if
    // it can make it more performant
    unsigned char* _device_img;
};
} // namespace popsift
