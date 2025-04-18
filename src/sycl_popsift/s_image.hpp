#pragma once

#include "sycl/event.hpp" // not sure if I need to include the event specifically when including sycl

#include <sycl/sycl.hpp>

namespace popsift {

struct Image
{
    Image() = delete;

    Image(sycl::queue Q);

    /** Create a device-sided buffer of the given dimensions */
    Image(int w, int h, sycl::queue Q);

    ~Image();

    void resetDimensions(int w, int h, float upscaleFactor);
    // void resetDimensions(int w, int h, int init_w, int init_h);
    void allocate_bindless();
    void allocate_usm(const float upscaleFactor);

    sycl::event copy_src_dev(unsigned char* input) { return _device_queue.memcpy(_device_src_img, input, _w * _h); }

    sycl::event load(void* input);
    // Need to fix these thre similar to th way I did load_liear by using the copy_src_dev before and use that value in
    // funciton
    sycl::event load_divide(unsigned char* input);
    sycl::event load_divide_point(unsigned char* input, const int& scaled_w);
    sycl::event load_divide_linear(unsigned char* input, const int& scaled_w);
    sycl::event load_linear(sycl::event src_img_transfer);
    // TODO: Should only be load_linear and load_point

    inline int getWidth() const { return _w; }
    inline int getHeight() const { return _h; }

    // Simple function for testing and development
    sycl::event host_move(void* outptu);
    void print_region(int start_x, int start_y, int end_x, int end_y);

    inline float* getInput() { return _device_img; }

  protected:
    int _w;        // width  of current image
    int _h;        // height of current image
    int _scaled_w; // scaled width
    int _scaled_h; // scaled height
    int _max_w;    // allocated width  of image
    int _max_h;    // allocated height of image
    int _init_w;   // Host image widht -- could be differnet due to apply scale factor done early
    int _init_h;   // host image height

    sycl::queue _device_queue;

    // TODO: Should demplate the class and only use _device_img;
    sycl::ext::oneapi::experimental::image_mem_handle _dev_img_handle;             // Underlying meory
    sycl::ext::oneapi::experimental::sampled_image_handle _sampled_dev_img_handle; // read only handle

    float* _device_img;
    unsigned char* _device_src_img;
};
} // namespace popsift
