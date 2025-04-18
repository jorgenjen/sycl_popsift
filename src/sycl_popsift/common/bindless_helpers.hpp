#pragma once

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/queue.hpp>

namespace popsift {
namespace sycl_bindless {

namespace syclexp = sycl::ext::oneapi::experimental;

// Probe whether the bindless API expects sampler before desc
// Added this due to the documentation saying
//
//   sampled_image_handle create_image(image_mem_handle memHandle,
//                                     const image_descriptor &desc,
//                                     const bindless_image_sampler &sampler,
//                                     const sycl::queue &syclQueue);
//
// Using it gave me an error. When I checked the bindless_images.hpp file I could not find this overload but found:
//
//      __SYCL_EXPORT sampled_image_handle
//      create_image(image_mem_handle memHandle, const bindless_image_sampler &sampler,
//                   const image_descriptor &desc, const sycl::queue &syclQueue);
//
// Hence I added this function that would work with both versions (done at compile time)

namespace defer {

// Add template to defer the resolution to avoid compilation error on the non existing overload
// Now overload resolution happens after if constexpr removes the branch that is not taken

// adding to namespace so it's clear it's not part of the usefull functions

// Q is always sycl::queue and no other type supported (only here to avoid compilation error)
template<typename Q>
syclexp::sampled_image_handle call_create_sampler_first(syclexp::image_mem_handle memHandle,
                                                        const syclexp::bindless_image_sampler& sampler,
                                                        const syclexp::image_descriptor& desc,
                                                        const Q& queue)
{
    return syclexp::create_image(memHandle, sampler, desc, queue);
}

template<typename Q>
syclexp::sampled_image_handle call_create_desc_first(syclexp::image_mem_handle memHandle,
                                                     const syclexp::bindless_image_sampler& sampler,
                                                     const syclexp::image_descriptor& desc,
                                                     const Q& queue)
{
    return syclexp::create_image(memHandle, desc, sampler, queue);
}

} // namespace defer

template<typename = void>
struct create_image_uses_sampler_first : std::false_type
{};

// If this overload exist it will become std::treu_type
template<>
struct create_image_uses_sampler_first<
  std::void_t<decltype(syclexp::create_image(std::declval<syclexp::image_mem_handle>(),
                                             std::declval<const syclexp::bindless_image_sampler&>(),
                                             std::declval<const syclexp::image_descriptor&>(),
                                             std::declval<const sycl::queue&>()))>> : std::true_type
{};

inline syclexp::sampled_image_handle create_sampled_image(syclexp::image_mem_handle memHandle,
                                                          const syclexp::bindless_image_sampler& sampler,
                                                          const syclexp::image_descriptor& desc,
                                                          const sycl::queue& queue)
{
    if constexpr(create_image_uses_sampler_first<>::value)
    {
        // What I found in my local installation
        return defer::call_create_sampler_first(memHandle, sampler, desc, queue);
    }
    else
    {
        // As statet in the documentation and in the example in the blog post
        // https://codeplay.com/portal/blogs/2025/02/11/sycl-bindless-images
        return defer::call_create_desc_first(memHandle, sampler, desc, queue);
    }
}

} // namespace sycl_bindless
} // namespace popsift
