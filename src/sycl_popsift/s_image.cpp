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

// Image::Image(int w, int h, sycl::queue Q)
//   : ImageBase(w, h, Q)
// {
//     // Not sure if using w and h is correct need to refactor the whole scaling thing as it is kinda confusing
//     // Should probably just use the scaled size and nothing else when using USM
//     _device_src_img = popsift::sycl_common::malloc_devT<unsigned char>(
//       w * h, __FILE__, __LINE__, "Could not allocate memory for image on device", Q);
//
//     _device_img = popsift::sycl_common::malloc_devT<float>(
//       w * h, __FILE__, __LINE__, "Could not allocate memory for float representation of image on device", Q);
// }

Image::~Image()
{
    fprintf(stderr, "\n\tDESTROYING IMAGE\n");
    if(_max_w == 0)
        return;

    sycl::free(_device_img, _device_queue);
    sycl::free(_device_src_img, _device_queue);
}

sycl::event Image::load(void* input)
{
    sycl::event src_img_transfer = copy_src_dev(input);

    return load_linear(src_img_transfer);
}

// Modified using sclaed and not scaled a bit confusing and ugly so should  refator if this is part of final
// void Image::resetDimensions(int w, int h, int scaled_w, int scaled_h)
void Image::resetDimensions(int w, int h, float upscaleFactor)
{
    if(_max_w == 0 && _max_h == 0)
    {
        // First time instantiating
        _max_w = _w = w;
        _max_h = _h = h;

        // if constexpr(sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>())
        // {
        //     // upscaleFactor is unused in this case
        //     allocate_bindless();
        //     allocate_usm(upscaleFactor); // neded for now as bindless not readly
        // }
        // else
        // {
        //     allocate_usm(upscaleFactor);
        // }
        allocate(upscaleFactor);

        return;
    }

    if(w == _w && h == _h)
        return;
    /* everything OK, nothing to do */

    if(w * h <= _max_w * _max_h)
    {
        // if constexpr(sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>())
        // {
        //     // Need to destroy and make again
        //     fprintf(stderr, "Need to deal with this realloc stufus\n");
        //     // Could all be done below
        // }
        // NOTE: wrap the return in this if ^ with negation ! and it only runs when using usm
        // else {}
        // TODO: Move to else when doing with texture meme version
        // smaller than current allocated segment hence it is fine to reuse in case of USM
        _w = w;
        _h = h;
        return;
    }

    // larger than current segment hence need to free and re-malloc

    sycl::free(_device_img, _device_queue);
    sycl::free(_device_src_img, _device_queue);

    _max_w = _w = w;
    _max_h = _h = h;

    _device_src_img = popsift::sycl_common::malloc_devT<unsigned char>(
      _scaled_w * _scaled_h, __FILE__, __LINE__, "Could not allocate memory for image on device", _device_queue);

    _device_img =
      popsift::sycl_common::malloc_devT<float>(_scaled_w * _scaled_h,
                                               __FILE__,
                                               __LINE__,
                                               "Could not allocate memory for float representation of image on device",
                                               _device_queue);
}

void Image::allocate(const float upscaleFactor)
{
    float scaleFactor = 1.0f / powf(2.0f, -upscaleFactor);

    _scaled_w = ceilf(_w * scaleFactor);
    _scaled_h = ceilf(_h * scaleFactor);

    _device_src_img = popsift::sycl_common::malloc_devT<unsigned char>(
      _scaled_w * _scaled_h, __FILE__, __LINE__, "Could not allocate memory for image on device", _device_queue);

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
    return _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(src_img_transfer);
        auto img = _device_img;
        auto input = _device_src_img;
        auto width = _w;
        auto height = _h;
        int step = _scaled_w / width; // floored -- not sure if it is corretc for other than 1 and 2
        int scaled_w = _scaled_w;
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

ImageBindless::ImageBindless(sycl::queue Q)
  : ImageBase(Q)
  , _dev_img_desc(syclexp::image_descriptor({0, 0}, 1, sycl::image_channel_type::unsigned_int8))
{}

ImageBindless::ImageBindless(int w, int h, sycl::queue Q)
  : ImageBase(w, h, Q)
  , _dev_img_desc(syclexp::image_descriptor({0, 0}, 1, sycl::image_channel_type::unsigned_int8))
{
    allocate();
}

ImageBindless::~ImageBindless()
{
    fprintf(stderr, "\n\tDESTROYING IMAGE\n");

    // Free bindless image
    // namespace syclexp = sycl::ext::oneapi::experimental;
    syclexp::free_image_mem(_dev_img_mem, syclexp::image_type::standard, _device_queue);
    syclexp::destroy_image_handle(_sampled_dev_img_handle, _device_queue);
}

sycl::event ImageBindless::load(void* input)
{
    // POP_FATAL("Currently not implemented\n");
    return _device_queue.ext_oneapi_copy(input, _dev_img_mem, _dev_img_desc);
}

void ImageBindless::resetDimensions(int w, int h, float /*upscaleFactor*/)
{
    if(_max_w == 0 && _max_h == 0)
    {
        // First time instantiating
        _max_w = _w = w;
        _max_h = _h = h;

        allocate();
        return;
    }

    if(w == _w && h == _h)
        return;
    /* everything OK, nothing to do */

    if(w * h <= _max_w * _max_h)
    {
        fprintf(stderr, "Need to deal with this realloc stufus\n");

        _w = w;
        _h = h;
        return;
    }

    // larger than current segment hence need to free and re-malloc

    // sycl::free(_device_img, _device_queue);
    // sycl::free(_device_src_img, _device_queue);
    //
    // _max_w = _w = w;
    // _max_h = _h = h;
    //
    // _device_src_img = popsift::sycl_common::malloc_devT<unsigned char>(
    //   _scaled_w * _scaled_h, __FILE__, __LINE__, "Could not allocate memory for image on device", _device_queue);
    //
    // _device_img =
    //   popsift::sycl_common::malloc_devT<float>(_scaled_w * _scaled_h,
    //                                            __FILE__,
    //                                            __LINE__,
    //                                            "Could not allocate memory for float representation of image on
    //                                            device", _device_queue);
}

void ImageBindless::allocate()
{
    //  Using bindless for input image (and scaling up with interpolation)

    fprintf(stderr, "Width and height used for bindless image w=%d h=%d", _w, _h);
    // namespace syclexp = sycl::ext::oneapi::experimental;

    _dev_img_desc.width = _w;
    _dev_img_desc.height = _h;

    // Should remove and use _dev_img_desc
    // syclexp::image_descriptor img_desc(
    //   {static_cast<size_t>(_w), static_cast<size_t>(_h)}, 1, sycl::image_channel_type::unsigned_int8);

    // Normalized 0-1 indexing (for upscaling accessing inbetween pixels in horiz)
    // Uses linear interpolation (hardware accelerated it should be)
    syclexp::bindless_image_sampler img_sampler(
      sycl::addressing_mode::repeat, sycl::coordinate_normalization_mode::normalized, sycl::filtering_mode::linear);

    // Not using RAII version for clear destroy and alloc (usefull for resizing)
    _dev_img_mem = syclexp::alloc_image_mem(_dev_img_desc, _device_queue);

    // Uses wrapper function to supprt two overloads (due to documentation and my installation being different)
    _sampled_dev_img_handle =
      popsift::sycl_bindless::create_sampled_image(_dev_img_mem, img_sampler, _dev_img_desc, _device_queue);

    // Use this free function for
    // void free_image_mem(image_mem_handle memHandle,
    //                     image_type imageType,
    //                     const sycl::queue &syclQueue);

    //     enum class image_channel_type : /* unspecified */ {
    //   snorm_int8,
    //   snorm_int16,
    //   unorm_int8,
    //   unorm_int16,
    //   signed_int8,
    //   signed_int16,
    //   signed_int32,
    //   unsigned_int8,
    //   unsigned_int16,
    //   unsigned_int32,
    //   fp16,
    //   fp32,
    // };
    //
    // enum class image_type : /* unspecified */ {
    //   standard,
    //   mipmap,
    //   array,
    //   cubemap,
    // };
}

} // namespace popsift
