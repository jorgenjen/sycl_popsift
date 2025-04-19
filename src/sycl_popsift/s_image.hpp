#pragma once

#include "sycl/event.hpp" // not sure if I need to include the event specifically when including sycl
#include "sycl_popsift/common/debug_macros.hpp"

#include <sycl/sycl.hpp>

#include <variant>

namespace popsift {

struct ImageBase
{
    ImageBase() = delete;

    ImageBase(sycl::queue Q);

    /** Create a device-sided buffer of the given dimensions */
    ImageBase(int w, int h, sycl::queue Q);

    // ~ImageBase();
    virtual ~ImageBase() = default;

    virtual void resetDimensions(int w, int h, float upscaleFactor) = 0;

    virtual sycl::event load(void* input) = 0;

    // Child overrides the one it uses
    virtual inline float* getInputFloat() { return nullptr; }
    virtual inline sycl::ext::oneapi::experimental::sampled_image_handle& getInputImage()
    {
        POP_FATAL("Not Implemented");
    }

    inline int getWidth() const { return _w; }
    inline int getHeight() const { return _h; }

  private:
    // virtual void allocate(const float upscaleFactor) = 0;
    // void allocate_usm(const float upscaleFactor);

  protected:
    int _w;        // width  of current image
    int _h;        // height of current image
    int _scaled_w; // scaled width
    int _scaled_h; // scaled height
    int _max_w;    // allocated width  of image
    int _max_h;    // allocated height of image
    // int _init_w;   // Host image widht -- could be differnet due to apply scale factor done early
    // int _init_h;   // host image height

    sycl::queue _device_queue;

    // TODO: Should demplate the class and only use _device_img;
    // Or mby make it a child class and the rest a base class
};

// Uses USM
struct Image : public ImageBase
{
    Image() = delete;
    Image(sycl::queue Q);
    Image(int w, int h, sycl::queue Q, const float upscaleFactor);
    ~Image() override;

    void resetDimensions(int w, int h, float upscaleFactor) override;

    // sycl::event copy_src_dev(unsigned char* input) { return _device_queue.memcpy(_device_src_img, input, _w * _h); }
    sycl::event copy_src_dev(void* input) { return _device_queue.memcpy(_device_src_img, input, _w * _h); }

    // Would allow this to be implemented for both float and char images
    sycl::event load(void* input) override;

    // sycl::event load(void* input);
    // Need to fix these thre similar to th way I did load_liear by using the copy_src_dev before and use that value
    // in funciton sycl::event load_divide(unsigned char* input); sycl::event load_divide_point(unsigned char* input,
    // const int& scaled_w); sycl::event load_divide_linear(unsigned char* input, const int& scaled_w);
    sycl::event load_linear(sycl::event src_img_transfer);
    // TODO: Should only be load_linear and load_point

    inline float* getInputFloat() override { return _device_img; }

  private:
    // void allocate(const float upscaleFactor) override;
    void allocate(const float upscaleFactor);

  protected:
    float* _device_img;
    unsigned char* _device_src_img;
};

// For bindless image
struct ImageBindless : public ImageBase
{
    ImageBindless() = delete;
    ImageBindless(sycl::queue Q);
    ImageBindless(int w, int h, sycl::queue Q);

    ~ImageBindless() override;

    void resetDimensions(int w, int h, float /*upscaleFactor*/) override;

    sycl::event load(void* input) override;

    inline sycl::ext::oneapi::experimental::sampled_image_handle& getInputImage() override
    {
        return _sampled_dev_img_handle;
    }

  private:
    // Ignore upscaleFactor for bindless as it does not use it
    // void allocate(const float /*upscaleFactor*/) override;
    void allocate();

  protected:
    sycl::ext::oneapi::experimental::image_descriptor _dev_img_desc;               // Descriptor for image and handle
    sycl::ext::oneapi::experimental::image_mem_handle _dev_img_handle;             // Underlying meory
    sycl::ext::oneapi::experimental::sampled_image_handle _sampled_dev_img_handle; // read only handle
};

} // namespace popsift
