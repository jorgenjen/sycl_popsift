#include "s_image.hpp"

#include "common/bindless_helpers.hpp"
#include "common/debug_macros.hpp"

#include <sycl/sycl.hpp>

#include <cmath> // ceilf
#include <cstdio>
#include <iostream>

namespace popsift {

// Alias used for bindlessimages
namespace syclexp = sycl::ext::oneapi::experimental;

/*************************************************************
 * ImageBase
 *************************************************************/

ImageBase::ImageBase(sycl::queue Q)
  : _w(0)
  , _h(0)
  , _max_w(0)
  , _max_h(0)
  , _device_queue(Q)
{}

ImageBase::ImageBase(int w, int h, sycl::queue Q)
  : _w(w)
  , _h(h)
  , _max_w(w)
  , _max_h(h)
  , _device_queue(Q)
{}

/*************************************************************
 * Image
 *************************************************************/

Image::Image(sycl::queue Q)
  : ImageBase(Q)
{}

Image::Image(int w, int h, sycl::queue Q, const float upscaleFactor)
  : ImageBase(w, h, Q)
{
    allocate(upscaleFactor);
}

Image::~Image()
{
    if(_max_w == 0)
        return;

    sycl::free(_device_img, _device_queue);
    sycl::free(_device_src_img, _device_queue);
}

sycl::event Image::load(void* input)
{
    sycl::event src_img_transfer = _device_queue.memcpy(_device_src_img, input, _w * _h);

    return load_linear(src_img_transfer);
}

// Can only be called in this cpp file
inline void Image::setScaledDims(const float upscaleFactor)
{
    float scaleFactor = 1.0f / powf(2.0f, -upscaleFactor);

    _scaled_w = ceilf(_w * scaleFactor);
    _scaled_h = ceilf(_h * scaleFactor);
}

// Modified using sclaed and not scaled a bit confusing and ugly so should  refator if this is part of final
// void Image::resetDimensions(int w, int h, int scaled_w, int scaled_h)
void Image::resetDimensions(int w, int h, float upscaleFactor)
{
    if(_max_w == 0 && _max_h == 0)
    {
        // First time instantiating
        _max_w = _w = w; // Stay unscaled
        _max_h = _h = h;

        allocate(upscaleFactor);

        return;
    }

    if(w == _w && h == _h)
        return;
    /* everything OK, nothing to do */

    if(w * h <= _max_w * _max_h)
    {
        // smaller than current allocated segment hence it is fine to reuse in this object as its USM
        // Could consider freeing and reallocing to make memory fotprint smaller as it will now stay at
        // The largest image used in lifetime of PopSift object
        _w = w;
        _h = h;
        setScaledDims(upscaleFactor);
        return;
    }

    // larger than current segment hence need to free and re-malloc

    sycl::free(_device_img, _device_queue);
    sycl::free(_device_src_img, _device_queue);

    _max_w = _w = w;
    _max_h = _h = h;

    allocate(upscaleFactor);
}

void Image::allocate(const float upscaleFactor)
{
    setScaledDims(upscaleFactor);

    // The source image that we use to compute the upscaled image
    _device_src_img = popsift::sycl_common::malloc_devT<unsigned char>(
      _w * _h, __FILE__, __LINE__, "Could not allocate memory for image on device", _device_queue);

    // The upscaled image
    _device_img =
      popsift::sycl_common::malloc_devT<float>(_scaled_w * _scaled_h,
                                               __FILE__,
                                               __LINE__,
                                               "Could not allocate memory for float representation of image on device",
                                               _device_queue);
}

// Only valid of the load functions others pass a host pointer to use which don't work on gpu need to transfer the image
// to device first before kernel launch
sycl::event Image::load_linear(sycl::event src_img_transfer)
{
    return _device_queue.submit(
      [&, img = _device_img, input = _device_src_img, width = _w, height = _h, scaled_w = _scaled_w](
        sycl::handler& cgh) {
          cgh.depends_on(src_img_transfer);
          int step = scaled_w / width; // floored -- not sure if it is corretc for other than 1 and 2
          cgh.parallel_for(sycl::range<2>(width, height), [=](sycl::id<2> idx) {
              auto in_pos = idx[0] + idx[1] * width;

              auto in_pos_right = idx[0] == width - 1 ? in_pos : in_pos + 1;
              auto in_pos_down = idx[1] == height - 1 ? in_pos : in_pos + width;
              auto in_pos_down_right = (idx[0] == width - 1 && idx[1] == height - 1) ? in_pos
                                       : idx[0] == width - 1                         ? in_pos + width
                                       : idx[1] == height - 1                        ? in_pos + 1
                                                              : in_pos + width + 1; // default case

              float pixel = static_cast<float>(input[in_pos]);
              float pixel_right = static_cast<float>(input[in_pos_right]);
              float pixel_down = static_cast<float>(input[in_pos_down]);
              float pixel_down_right = static_cast<float>(input[in_pos_down_right]);

              auto pos = idx[0] * step + idx[1] * step * scaled_w; // position in potentially upscaled image

              img[pos] = pixel;
              img[pos + 1] = (pixel + pixel_right) / 2;
              img[pos + scaled_w] = (pixel + pixel_down) / 2;
              img[pos + scaled_w + 1] = (pixel + pixel_down + pixel_right + pixel_down_right) / 4;
          });
      });
}

/*************************************************************
 * ImageBindless
 *************************************************************/

// Wheter or not to use copy to aligned memory before copy to bindless image mem
// Could result in better transfer speeds
#define USE_ALIGNED_STAGING true
ImageBindless::ImageBindless(sycl::queue Q)
  : ImageBase(Q)
  , _dev_img_desc(syclexp::image_descriptor({0, 0}, 1, sycl::image_channel_type::unorm_int8))
{}

ImageBindless::ImageBindless(int w, int h, sycl::queue Q)
  : ImageBase(w, h, Q)
  , _dev_img_desc(syclexp::image_descriptor({0, 0}, 1, sycl::image_channel_type::unorm_int8))
{
    allocate<true>();
}

template<bool freeAlignedHost>
void ImageBindless::free()
{
    try
    {
        if constexpr(USE_ALIGNED_STAGING && freeAlignedHost)
        {
            if(_aligned_src_img)
            {
                sycl::free(_aligned_src_img, _device_queue);
                _aligned_src_img = nullptr;
            }
        }

        if(_sampled_handle_created)
        {
            // Destroy handle before underlying memory structure
            syclexp::destroy_image_handle(_sampled_dev_img_handle, _device_queue);
        }

        if(_img_mem_allocated)
        {
            syclexp::free_image_mem(_dev_img_mem, syclexp::image_type::standard, _device_queue);
        }
    }
    catch(sycl::exception& e)
    {
        std::stringstream ss;
        ss << "SYCL exception caught in BindlessImage destructor: " << e.what();
        POP_FATAL(ss.str());
    }
    catch(std::exception& e)
    {
        std::stringstream ss;
        ss << "std exception caught in BindlessImage destructor:" << e.what();
        POP_FATAL(ss.str());
    }
    catch(...)
    {
        POP_FATAL("Caught unknown exception in BindlessImage destructor")
    }
}

sycl::event ImageBindless::load(void* input)
{
#if USE_ALIGNED_STAGING
    sycl::event copy_to_aligned = _device_queue.memcpy(_aligned_src_img, input, _w * _h);

    return _device_queue.ext_oneapi_copy(_aligned_src_img, _dev_img_mem, _dev_img_desc, copy_to_aligned);
#else
    return _device_queue.ext_oneapi_copy(input, _dev_img_mem, _dev_img_desc);
#endif
}

void ImageBindless::resetDimensions(int w, int h, float /*upscaleFactor*/)
{
    if(_max_w == 0 && _max_h == 0)
    {
        // First time instantiating
        _max_w = _w = w;
        _max_h = _h = h;

        allocate<true>();
        return;
    }

    if(w == _w && h == _h)
        return;
    /* everything OK, nothing to do */

    if(w * h <= _max_w * _max_h)
    {
        // Can keep _aligned_src_img as it's segment is larger than needed
        // Need to free/destroy bindless image still
        _w = w;
        _h = h;
        free<false>();
        allocate<false>();
        return;
    }

    // larger than current segment hence need to free and re-malloc
    _max_w = _w = w;
    _max_h = _h = h;
    free<true>();
    allocate<true>();
}

template<bool allocAlignedHost>
void ImageBindless::allocate()
{
    // Rest of desc was set in constructor -- Updating widht and height
    _dev_img_desc.width = _w;
    _dev_img_desc.height = _h;

    // Normalized 0-1 indexing (for upscaling accessing inbetween pixels in horiz)
    // Uses linear interpolation (hardware accelerated it should be)
    syclexp::bindless_image_sampler img_sampler(sycl::addressing_mode::clamp_to_edge,
                                                sycl::coordinate_normalization_mode::normalized,
                                                sycl::filtering_mode::linear);

    if constexpr(USE_ALIGNED_STAGING && allocAlignedHost)
    {
        _aligned_src_img = sycl::aligned_alloc_host(128, _w * _h, _device_queue);
        if(!_aligned_src_img)
            POP_FATAL("Failed to allocate aligned host memory");
    }

    try
    {
        // Not using RAII version for clear destroy and alloc (usefull for resizing idk if you can with RAII)
        _dev_img_mem = syclexp::alloc_image_mem(_dev_img_desc, _device_queue);
        _img_mem_allocated = true;

        // Uses wrapper function to supprt two overloads (due to documentation and my installation being different)
        _sampled_dev_img_handle =
          popsift::sycl_bindless::create_sampled_image(_dev_img_mem, img_sampler, _dev_img_desc, _device_queue);
        _sampled_handle_created = true;

        // set flags based on function before it did not throw hence it must been allocated/created
    }
    catch(sycl::exception& e)
    {
        std::stringstream ss;
        ss << "SYCL exception caught in allocate method in BindlessImage: " << e.what();
        POP_FATAL(ss.str());
    }
    catch(std::exception& e)
    {
        std::stringstream ss;
        ss << "std exception caught in allocate method in BindlessImage: " << e.what();
        POP_FATAL(ss.str());
    }
    catch(...)
    {
        POP_FATAL("Caught unknown exception in allocate method in BindlessImage")
    }
}

} // namespace popsift
