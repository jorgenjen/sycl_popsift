#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/persistent_config_macros.h" // If we are using root group or handcrafted wg syncrinozation
#include "sycl_popsift/persistent_configuration.hpp"
#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

namespace popsift {

template<bool REMAINDER_COL>
static inline void horiz_bindless_input(float* intermediate,
                                        syclexp::sampled_image_handle src,
                                        const float* filter,
                                        const int span,
                                        const int dst_w,
                                        const int write_x,
                                        int write_y,
                                        float read_x,
                                        float read_y,
                                        int base_pos)
{
    if constexpr(REMAINDER_COL)
    {
        if(write_x >= dst_w)
            return;
    }

    float out = 0.0f;

    // #pragma unroll
    //     for(int offset = span; offset > 0; offset--)
    //     {
    //         const float g = filter[offset];
    //         const float offrel = float(offset) / dst_w; // relative offset
    //         const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
    //         const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
    //         out += ((v1 + v2) * g);
    //     }
    //
    //     const float& g = filter[0];
    //     const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    //     out += (v3 * g);
    //
    //     // if(write_x < 120 && write_x > 110 && write_y < 120 && write_y > 110)
    //     if(write_x < 900 && write_x > 890 && write_y < 500 && write_y > 490)
    //     {
    //         syclexp::printf("write(%d, %d) --> out = %f -- read_x = %f - read_y = %f -- wave bindless\n",
    //                         write_x,
    //                         write_y,
    //                         out,
    //                         read_x,
    //                         read_y);
    //     }
    //
    //     intermediate[write_x + write_y * dst_w] = out * 255.0f;

#if true

#pragma unroll
    for(int offset = span; offset > 0; offset--)
    {
        const float g = filter[offset];
        const float offrel = float(offset) / dst_w; // relative offset
        const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
        const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
        out += ((v1 + v2) * g);

        // const float v1 = buffer[base_pos - offset];
        // const float v2 = buffer[base_pos + offset];
        // out += ((v1 + v2) * g);
    }

    const float& g = filter[0];
    const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    out += (v3 * g);
    // out += (buffer[base_pos] * filter[0]);

    // if(write_x < 120 && write_x > 110 && write_y < 120 && write_y > 110)
    // if(write_x < 900 && write_x > 890 && write_y < 500 && write_y > 490)
    // {
    //     syclexp::printf("write(%d, %d) --> out = %f -- read_x = %f - read_y = %f wave bindless\n",
    //                     write_x,
    //                     write_y,
    //                     out,
    //                     read_x,
    //                     read_y);
    // }

    intermediate[write_x + write_y * dst_w] = out * 255.0f;
#endif

    // #pragma unroll
    //     for(int offset = span; offset > 1; offset--)
    //     {
    //         const float g = filter[offset];
    //         const float offrel = float(offset) / dst_w; // relative offset
    //         const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
    //         const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
    //         out += ((v1 + v2) * g);
    //
    //         // const float v1 = buffer[base_pos - offset];
    //         // const float v2 = buffer[base_pos + offset];
    //         // out += ((v1 + v2) * g);
    //     }
    //
    //     const float& g = filter[0];
    //     const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    //     out += (v3 * g);
    //     // out += (buffer[base_pos] * filter[0]);
    //
    //     // if(write_x < 120 && write_x > 110 && write_y < 120 && write_y > 110)
    //     if(write_x < 900 && write_x > 890 && write_y < 500 && write_y > 490)
    //     {
    //         syclexp::printf("write(%d, %d) --> out = %f -- read_x = %f - read_y = %f wave bindless\n",
    //                         write_x,
    //                         write_y,
    //                         out,
    //                         read_x,
    //                         read_y);
    //     }
    //
    //     intermediate[write_x + write_y * dst_w] = out * 255.0f;
}

template<bool REMAINDER_COL>
static inline void horiz_local_mem(float* intermediate,
                                   sycl::local_accessor<float, 1> buffer,
                                   const float* filter,
                                   const int span,
                                   const int dst_w,
                                   const int write_x,
                                   int write_y,
                                   int base_pos)
{
    if constexpr(REMAINDER_COL)
    {
        if(write_x >= dst_w)
            return;
    }

    float out = 0.0f;

#pragma unroll
    for(int offset = span; offset > 0; offset--)
    {
        const float g = filter[offset];
        // const float offrel = float(offset) / dst_w; // relative offset
        // const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
        // const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
        // out += ((v1 + v2) * g);

        const float v1 = buffer[base_pos - offset];
        const float v2 = buffer[base_pos + offset];
        out += ((v1 + v2) * g);
    }

    // const float& g = filter[0];
    // const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    // out += (v3 * g);
    out += (buffer[base_pos] * filter[0]);

    // if(write_x < 120 && write_x > 110 && write_y < 120 && write_y > 110)
    // if(write_x < 900 && write_x > 890 && write_y < 500 && write_y > 490)
    // {
    //     syclexp::printf("write(%d, %d) --> out = %f -- wave local mem\n", write_x, write_y, out);
    // }

    intermediate[write_x + write_y * dst_w] = out * 255.0f;
}

static inline void vert_persistent(float* intermediate,
                                   sycl::local_accessor<float, 1> buffer,
                                   const float* filter,
                                   const int span,
                                   const int dst_w,
                                   const int write_x,
                                   int write_y,
                                   int base_pos)
{
    // Vert kernel that resues registers
}

#define USE_ATOMIC_SYNC 1 // For using atomic ref on the work-group state used for synchronizatio

// synchronizes vert execution so that ll data needed to do horiz is available and correct
static inline void vert_sync_for_horiz(int* wg_sync_state, sycl::nd_item<2>& it, int wait_on_state)
{
#if USE_ROOT_GROUP
    sycl::group_barrier(root);
#else
    // Use local hand crafted sychronization
    sycl::group group = it.get_group();
    sycl::group_barrier(group); // Ensure all have done horiz

    if(it.get_local_linear_id() == 0) // only one per work_group
    {
        // Convert to int to use less registers (might help)
        int group_pos = it.get_group(1);
        int group_final_index = it.get_group_range(1) - 1;
        int group_linear_pos = it.get_group_linear_id();
#if USE_ATOMIC_SYNC
        // Need to be a 4 byte wide data type like int or unsigned int for this to work...
        sycl::atomic_ref<int,
                         sycl::memory_order_relaxed,
                         sycl::memory_scope_device,
                         sycl::access::address_space::global_space>(wg_sync_state[group_linear_pos])++;

        // Active wait-- spin lock
        if(group_pos == 0)
        {
            // left border -- only depends on right

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              right(wg_sync_state[group_linear_pos + 1]);

            while(right < wait_on_state) {}
        }
        else if(group_pos == group_final_index)
        {
            // right most border -- only depends on left

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              left(wg_sync_state[group_linear_pos - 1]);
            while(left < wait_on_state) {}
        }
        else
        {
            // Normal in the middle  -- depends on left and right
            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              left(wg_sync_state[group_linear_pos - 1]);

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              right(wg_sync_state[group_linear_pos + 1]);

            while(left < wait_on_state && right < wait_on_state) {}
        }

#else
        // BUG: This deadlocks probably does not get the update and the value is stale
        // --> Seems like we need atomic might work with atomic for only read or write not sure seems risky
        // --> Using atomic for both with the one above instead of this one

        // Use normal memory think that should be fine for global aswell...
        // Could cause deadlock if it never let's other's work or if it does not properly update the value on
        // each iteration and just continues to read the initial stale value -- if so atomics would be required

        // int group_pos = it.get_group(1);
        // int group_final_index = it.get_group_range(1) - 1;
        // int group_linear_pos = it.get_group_linear_id();
        // work group leader
        wg_sync_state[group_linear_pos]++; // Signal horiz done

        // Active wait -- spin lock
        if(group_pos == 0)
        {
            // left border -- only depends on right
            while(wg_sync_state[group_linear_pos + 1] < 1) {}
        }
        else if(group_pos == group_final_index)
        {
            // right most border -- only depends on left
            while(wg_sync_state[group_linear_pos - 1] < 1) {}
        }
        else
        {
            // Normal in the middle  -- depends on left and right
            while(wg_sync_state[group_linear_pos - 1] < 1 && wg_sync_state[group_linear_pos + 1] < 1) {}
        }
#endif
    }
    sycl::group_barrier(group); // Wait for wg leader to finish spin lock ensuring dependencies are done
#endif
}

// synchronizes horiz execution so that all data needed to do vert is available and correct
static inline void horiz_sync_for_vert(int* wg_sync_state, sycl::nd_item<2>& it, int wait_on_state)
{
#if USE_ROOT_GROUP
    sycl::group_barrier(root);
#else
    // Use local hand crafted sychronization -- Volatile requires it all to be scheduled in one wave
    sycl::group group = it.get_group();
    sycl::group_barrier(group); // Ensure all have done horiz

    if(it.get_local_linear_id() == 0) // only one per work_group
    {
        // Convert to int to use less registers (might help)
        int group_pos_0 = it.get_group(0);
        int group_pos_1 = it.get_group(1);
        int group_range_0 = it.get_group_range(0);
        int group_range_1 = it.get_group_range(1);
        // int group_linear_pos = it.get_group_linear_id();
        // Need to be a 4 byte wide data type like int or unsigned int for this to work...
        sycl::atomic_ref<int,
                         sycl::memory_order_relaxed,
                         sycl::memory_scope_device,
                         sycl::access::address_space::global_space>(wg_sync_state[it.get_group_linear_id()])++;

        // Active wait-- spin lock
        if(group_pos_0 == 0)
        {
            // top border -- only depends on wg below

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              below(wg_sync_state[(group_pos_0 + 1) * group_range_1 + group_pos_1]);

            while(below < wait_on_state) {}
        }
        else if(group_pos_0 == group_range_0 - 1)
        {
            // right most border -- only depends on left

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              above(wg_sync_state[(group_pos_0 - 1) * group_range_1 + group_pos_1]);
            while(above < wait_on_state) {}
        }
        else
        {
            // Normal in the middle  -- depends on left and right
            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              above(wg_sync_state[(group_pos_0 - 1) * group_range_1 + group_pos_1]);

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              below(wg_sync_state[(group_pos_0 + 1) * group_range_1 + group_pos_1]);

            while(above < wait_on_state && below < wait_on_state) {}
        }
    }
    sycl::group_barrier(group); // Wait for wg leader to finish spin lock ensuring dependencies are done
#endif
}

namespace normalizedSource {

// Used for ImageBindless
// Only used on input image (initial)
// And only works for it due to  filter and span selection

// aspect::ext_oneapi_bindless_sampled_image_fetch_2d
// This aspect is required to use sampled image need to add a check for that earlier in selection
// template<bool if_required>

// For debugging remove!
#define DO_HORIZ 0
#define DO_VERT 1

#define USE_SHARED_MEM_FOR_INPUT 1
template<bool REMAINDER_COL, bool REMAINDER_ROW>
class BuildOctave
{
  private:
    syclexp::sampled_image_handle src;
    float** data_array; // Need to be array of all dst data
    float** dog_array;
    float* intermediate;
    popsift::GaussInfo* d_gauss;
    sycl::local_accessor<float, 1> buffer;
    const sg_region_blocks sg_region;
    const int dst_w;
    const int dst_h;
    const float shift;
    const int levels;

  public:
    BuildOctave(syclexp::sampled_image_handle src,
                float** data_array,
                float** dog_array,
                float* intermediate,
                popsift::GaussInfo* d_gauss,
                sycl::local_accessor<float, 1> buffer,
                const sg_region_blocks sg_region,
                const int dst_w,
                const int dst_h,
                const float shift,
                const int levels)

      : src(src)
      , data_array(data_array)
      , dog_array(dog_array)
      , intermediate(intermediate)
      , d_gauss(d_gauss)
      , buffer(buffer)
      , sg_region(sg_region)
      , dst_w(dst_w)
      , dst_h(dst_h)
      , shift(shift)
      , levels(levels) {};

    inline void operator()(sycl::nd_item<2> it) const
    {
        // const auto sg_width = it.get_sub_group().get_max_local_range()[0]; // 32 in cuda

        const int write_x = it.get_global_id(1);
        // int write_y = it.get_group(0) * sg_region.height; // Changes in normal block aswell
        int write_y = it.get_global_id(0) * sg_region.height; // Changes in normal block

#if DO_HORIZ
        // Used for input only
        const float* filter_input = &d_gauss->dd.filter[0];
        const int span_input = d_gauss->dd.span[0];

        const float read_x = (write_x + shift) / dst_w;
        float read_y = (write_y + shift) / dst_h;

        // Not sure if there is a point of using this for input level -- As we can't async load
        const int base_pos =
          (it.get_local_range(1) + (span_input << 1)) * (it.get_local_id(0) << 1) + it.get_local_id(1) + span_input;

        // Second buffer row (there are two per row in the work-group)
        const int base_pos_2 = (it.get_local_range(1) + (span_input << 1)) * ((it.get_local_id(0) << 1) + 1) +
                               it.get_local_id(1) + span_input;

        // const int rel_span = ((1 / dst_w) * span); // Relative span value used for offset
        const float rel_span = float(span_input) / dst_w; // Relative span value used for offset

        // for(int i = 0; i < sg_region.height; i++)

        // const float read_y_increment = 1.0f / dst_h; // Does not result in the same as recompute due to
        // accumulation of floating point error

        int loop_end = write_y + sg_region.height;

        if constexpr(REMAINDER_ROW)
        {
            if(loop_end >= dst_h)
                loop_end = dst_h; // Limit to last pixel
        }

#if USE_ROOT_GROUP
        auto root = it.ext_oneapi_get_root_group(); // Root group all work_items running kernel
#endif
        for(; write_y < loop_end; ++write_y) // Modifies write_y want that later
        {
            // read_y += read_y_increment; // Floating point error accumulation hence not using
            read_y = (write_y + shift) / dst_h;

#if USE_SHARED_MEM_FOR_INPUT
            buffer[base_pos] = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y}); // every one does this

            if(it.get_local_id(1) < span_input)
            {
                // load left side (lenght of span)
                buffer[base_pos - span_input] =
                  syclexp::sample_image<float>(src, sycl::float2{read_x - rel_span, read_y});
            }
            else if(it.get_local_id(1) >= (it.get_local_range(1) - span_input))
            {
                buffer[base_pos + span_input] =
                  syclexp::sample_image<float>(src, sycl::float2{read_x + rel_span, read_y});
            }

            // Here would be good to do async load of next row but does not seem to be possible to do with bindless
            // images But for remaining parts it will be not sure if we should use local mem for this part however
            sycl::group_barrier(it.get_group()); // Ensure all is loaded before we do horiz

            horiz_local_mem<REMAINDER_COL>(
              intermediate, buffer, filter_input, span_input, dst_w, write_x, write_y, base_pos);

#else
            horiz_bindless_input<REMAINDER_COL>(
              intermediate, src, filter, span, dst_w, write_x, write_y, read_x, read_y, base_pos);

#endif

            // Second row buffer in use: Same as above otherwise

            write_y++;
            if(write_y >= loop_end)
                break;

            // read_y += read_y_increment; // Floating point error accumulation hence not using
            read_y = (write_y + shift) / dst_h;

#if USE_SHARED_MEM_FOR_INPUT
            buffer[base_pos_2] = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});

            if(it.get_local_id(1) < span_input)
            {
                buffer[base_pos_2 - span_input] =
                  syclexp::sample_image<float>(src, sycl::float2{read_x - rel_span, read_y});
            }
            else if(it.get_local_id(1) >= (it.get_local_range(1) - span_input))
            {
                buffer[base_pos_2 + span_input] =
                  syclexp::sample_image<float>(src, sycl::float2{read_x + rel_span, read_y});
            }

            sycl::group_barrier(it.get_group());

            horiz_local_mem<REMAINDER_COL>(
              intermediate, buffer, filter_input, span_input, dst_w, write_x, write_y, base_pos_2);
#else
            horiz_bindless_input<REMAINDER_COL>(
              intermediate, src, filter_input, span_input, dst_w, write_x, write_y, read_x, read_y, base_pos);
#endif
        }
        // At end of this loop write_y will be equal to loop_end or smaller but it wil always be one too large hence
        // need to subtract one
        write_y--;

        // Synchronize and then do horiz
        horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);

        // #if false
        // Start doing Vert then later we do horiz on data_array so not using sampled image then we can use async
        // load of next row Do vert for this one then make loop over the levels for the rest with horiz from prev
        // and vert from intermediate

        // Vert
        sycl::group_barrier(it.get_group());

#endif
        // TODO:  Add tempalte and if constexpr to have different versions based on if horiz and vert are using shared
        // meme solution need non shared mem solution aswell to support that (don't think any GPU would not support
        // horiz as is now so only needed for vert I think)
#if DO_VERT
#if !DO_HORIZ
        // So it is the same it would have been if we were doing horiz
        write_y = sycl::min(write_y + sg_region.height - 1, dst_h - 1);
        if(it.get_global_linear_id() == 0)
        {
            syclexp::printf("write_y = %d intermediate_value = %f \n ", write_y, intermediate[write_y * dst_w]);
        }
#endif

        const auto row_width = [&]() {
            // could remove constexpr to avoid having so many templated classes which could increase load times
            if constexpr(REMAINDER_ROW)
            {
                if(it.get_group(1) == (it.get_group_range(1) - 1))
                {
                    // final wg column
                    // return (static_cast<int>(it.get_local_range(1) - it.get_global_range(1)) - dst_w);
                    return it.get_local_range(1) - (it.get_global_range(1) - dst_w);
                }
                // Rest of columns
                return it.get_local_range(1);
            }
            else
            {
                // There is no remainder hence all are full rows
                return it.get_local_range(1);
            }
        }();

        // need to ensure only it.get_local_id(1) < row_width is doing something but I think they also need to reach the
        // async work group copy for that to work So need if protection on all else code to avoid it messing that up

        // bottom of our region
        int start_pos = write_y * dst_w + it.get_group(1) * it.get_local_range(1);
        // Bottom row position
        auto intermediate_ptr =
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
            intermediate + start_pos);

        // auto inter =
        //   sycl::make_ptr<sycl::access::address_space::global_space, sycl::access::decorated::yes>(intermediate);
        // auto intermeidate_ptr = inter.template get_multi_ptr<sycl::access::decorated::yes>();
        auto buffer_ptr = buffer.template get_multi_ptr<sycl::access::decorated::yes>();

        // Changes with loop matching level -- initial is zero always
        int span = d_gauss->inc.span[0];

        sycl::group group = it.get_group();

        // NOTE:  Window is always as wide as wide as it.get_local_range(1) but we only load in row_widht wide data
        // This still alows for no bank conflicts and the work-items outside of range can safely do the same as rest on
        // the dono't care data in their location and we only need so safe guard the writes to global memory minimizing
        // the number of if checks we need (and doing more in locstep does not really add performance overhead)

        // Load into middle of window
        sycl::device_event evt_center =
          group.async_work_group_copy(buffer_ptr + (span * it.get_local_range(1)), intermediate_ptr, row_width);

        // Changes with loop matching level -- initial is zero always
        float* filter = &d_gauss->inc.filter[0];

        std::optional<sycl::device_event> above_events[2]; // Avoids using deleted default constructor of device_event
        std::optional<sycl::device_event> below_events[2]; // Avoids using deleted default constructor of device_event

        // if(write_y - 1 >= 0)
        // {
        // Above is always safe in this case as we are at the bottom of our region hence one above always exits

        // If true it should only do async_work_group copy and nothing else
        bool live = it.get_local_id(1) < row_width;
        // bool exclude = it.get_local_id(1) >= row_width;

        // Event is reused in loop
        // sycl::device_event above_1_evt =
        above_events[0] = group.async_work_group_copy(
          buffer_ptr + ((span - 1) * it.get_local_range(1)), intermediate_ptr - dst_w, row_width);
        // }

        evt_center.wait(); // Need to finish in case we need to copy from it for clamping below
        // At this point we could just do it synchronous...
        if(write_y + 1 < dst_h)
        {
            // below_1_evt =
            below_events[0] = group.async_work_group_copy(
              buffer_ptr + ((span + 1) * it.get_local_range(1)), intermediate_ptr + dst_w, row_width);
        }
        else
        {
            // Clamp to edge by copying from center to 1 below
            buffer[(span + 1) * it.get_local_range(1) + it.get_local_id(1)] =
              buffer[span * it.get_local_range(1) + it.get_local_id(1)];
            // No need for barrier as only work_item that reads from it is the one that wrote to it hence no need to
            // sync
        }

        // float val = buffer[span * row_width + it.get_local_id(1)];
        // float val = buffer[span * it.get_local_range(1) + it.get_local_id(1)];
        float val = 0.0f;
        // float out = val * filter[0];
        float out; // Need to add the self part last to get same result as popsift makes big difference on filnal
                   // values due to float inaccuracies

        float out_sep = val * filter[0];
        float g;

        if(it.get_global_linear_id() == 0)
        {
            // syclexp::printf("Val self = %f\n", val);
            syclexp::printf("Val self = %.10f FILTER=%.20f\n", val, filter[0]);
        }

        int i_max = dst_h - write_y - 1; // How many pixels we can go down before we are beyond image bounds
        int next_i;

        float val_above;
        float val_below;
        for(int i = 1; i <= span; ++i)
        {
            // int next_i = i + 1;
            next_i = i + 1;
            if(next_i <= span)
            {
                // Above is known to be safe here aslong as we keep mimimum height of block to largest_span + 1
                // As then we know it's withing bounds for top row and we can omit the check here (we do need it
                // later on when we start sliding the window up)
                above_events[1] = group.async_work_group_copy(
                  buffer_ptr + ((span - next_i) * it.get_local_range(1)), intermediate_ptr - dst_w * next_i, row_width);

                if(i <= i_max)
                {
                    below_events[1] =
                      group.async_work_group_copy(buffer_ptr + ((span + next_i) * it.get_local_range(1)),
                                                  intermediate_ptr + dst_w * next_i,
                                                  row_width);
                }

                // Could consider doing this to avoid the if below but might not be worth it
                // NOTE:  Dont' think this is safe due to the posibility for this event not to be populated and then
                // we wait on a non existing event not sure what heppens then SAfe when using std::optional and
                // using has_value() method to verify

                else
                {
                    // Copy from i_max position into it's row position so that we don't need to do the if check
                    // later on in the loop -- Does not happen for most Sub_groups so should be worth the tradeoff
                    // and local to local memcopy should not be that slow -- And as each work-item is only using the
                    // value they are copying we do not need to use a barrier to ensure all can see the update as
                    // only our-work item needs it (essentially a register spill hardcoded)

                    // Copy into our position the edge value from where it was stored in local memory (always safe)
                    buffer[(span - next_i) * it.get_local_range(1) + it.get_local_id(1)] =
                      buffer[(span - i_max) * it.get_local_range(1) + it.get_local_id(1)];
                }
            }
            // First iteration wait() always exists
            g = filter[i];

            above_events[0]->wait();
            val_above = buffer[(span - i) * it.get_local_range(1) + it.get_local_id(1)];
            // Could compute and add to out here but I think doing it all in one seems to be more
            // sensible right? But if we are properly memory bound doing it here makes more sense as the
            // extra multliplication don't hurt in that case and more is done earlier

            if(below_events[1].has_value())
            {
                below_events[1]->wait();
            }
            val_below = buffer[(span + i) * it.get_local_range(1) + it.get_local_id(1)];
            // Always in correct position due to copying in case of clamping

            // below_events[0]->wait();
            // // Clamp to edge
            // val_below = i <= i_max ? buffer[(span + i) * it.get_local_range(1) + it.get_local_id(1)]
            //                        : buffer[(span + i_max) * it.get_local_range(1) + it.get_local_id(1)];

            if(it.get_global_linear_id() == 0)
            {
                syclexp::printf(
                  "i=%d vals: Above=%.10f Below=%.10f FILTER = %.20f-- TOP IN LOOP\n ", i, val_above, val_below, g);
            }

            // Safe as notmal Vert (just smarter placed there)
            // out += (val_above * g);
            // out += (val_below * g);
            out += ((val_above + val_below) * g);

            out_sep += (val_above * g);
            out_sep += (val_below * g);

            // ######################################################################################################
            // Second iteration of the same just using different variables for events could do the same with array
            // of events and do mod to figoure out which one to use but this should be less expensive than doing mod
            // (though less readable)
            // ######################################################################################################

            // Next iteration
            if(next_i > span) // Exit check
                break;

            i = next_i; // Not sure if incrementing is faster ?
            next_i++;
            // Next iteration with swaped event variables

            if(next_i <= span)
            {
                above_events[0] = group.async_work_group_copy(
                  buffer_ptr + ((span - next_i) * it.get_local_range(1)), intermediate_ptr - dst_w * next_i, row_width);

                if(next_i <= i_max)
                {
                    above_events[0] =
                      group.async_work_group_copy(buffer_ptr + ((span + next_i) * it.get_local_range(1)),
                                                  intermediate_ptr + dst_w * next_i,
                                                  row_width);
                }
                else
                {
                    buffer[(span - next_i) * it.get_local_range(1) + it.get_local_id(1)] =
                      buffer[(span - i_max) * it.get_local_range(1) + it.get_local_id(1)];
                }
            }

            g = filter[i];

            above_events[1]->wait();
            val_above = buffer[(span - i) * it.get_local_range(1) + it.get_local_id(1)];

            // Don't need to ensure clamping as we have copied clamped values to their correct position in window to
            // avoid having to do if checks in loop below
            if(below_events[1].has_value())
            {
                below_events[1]->wait();
            }
            val_below = buffer[(span + i) * it.get_local_range(1) + it.get_local_id(1)];
            // else
            // {
            //     // Clamp to edge
            //     val_below = buffer[(span + i_max) * it.get_local_range(1) + it.get_local_id(1)];
            // }
            // val_below = i <= i_max ? buffer[(span + i) * it.get_local_range(1) + it.get_local_id(1)]
            //                        : buffer[(span + i_max) * it.get_local_range(1) + it.get_local_id(1)];

            if(it.get_global_linear_id() == 0)
            {
                syclexp::printf(
                  "i=%d vals: Above=%.10f Below=%.10f FIlTER = %.20f-- BOTTOM IN LOOP\n ", i, val_above, val_below, g);
            }

            // out += (val_above * g);
            // out += (val_below * g);
            out += ((val_above + val_below) * g);

            out_sep += (val_above * g);
            out_sep += (val_below * g);
            // Now we have done second iteration and next to wait is above and below 1 and prefetch 2 so we iterate
        }
        // Do we want to do write async aswell? Or will that just result in worse performance? mby test
        if(live)
        {
            if(it.get_global_linear_id() == 0)
            {
                syclexp::printf("Final value of out for self = %.10f -- out_sep = %.10f\n", out, out_sep);

                // Compute whole value print every step

#if true

                // int out; // Local so we don't modify outer out
                float out = 0.0f;
                float out_2 = 0.0f;
                float out_3 = 0.0f;
                float val;
                float val_2;
                // float val_3;
                float g;
                int x = 0;
                int y = 97;
                int height = dst_h;
                int width = dst_w;

                g = filter[0];

                val_2 = buffer[span * it.get_local_range(1) + it.get_local_id(1)];
                out_2 = (val_2 * g);

                syclexp::printf("SELF -- added_to_out_3= %.10f\n", (val_2 * g));

                for(int offset = span; offset > 0; offset--)
                // for(int offset = span - 1; offset > 0; offset--)
                {
                    g = filter[offset];

                    int idy = y - offset;
                    val = idy < 0 ? intermediate[x] : intermediate[x + idy * width]; // clamp edge
                    out += (val * g);

                    val_2 = buffer[(span - offset) * it.get_local_range(1) + it.get_local_id(1)];
                    out_2 += (val_2 * g);
                    out_3 += (val_2 * g); // Doing self increment last

                    syclexp::printf("offset=%d  vals: Above=%.10f -- out=%.10f --> val_2=%.10f - out_2=%.10f\n",
                                    offset,
                                    span,
                                    val,
                                    out,
                                    val_2,
                                    out_2);
                    // "offset = %d span =%d vals: Above=%f -- out = %f --> ", offset, span, val, out);

                    idy = y + offset;
                    val = idy >= height ? intermediate[x + (height - 1) * width]
                                        : intermediate[x + idy * width]; // clamp edge
                    out += (val * g);

                    val_2 = buffer[(span + offset) * it.get_local_range(1) + it.get_local_id(1)];
                    out_2 += (val_2 * g);
                    out_3 += (val_2 * g); // Doing self increment last

                    // if(x == 0 && y == 97 && level == 0)
                    // if(x == 0 && y == 97)
                    // {
                    // syclexp::printf("offset = %d vals: Above=%f Below=%f FILTER=%f\n", offset, val_1, val, g);
                    syclexp::printf("Below=%.10f -- out =%.10f FILTER=%.20f --> val_2=%.10f - out_2=%.10f\n\n",
                                    val,
                                    out,
                                    g,
                                    val_2,
                                    out_2);
                    // }
                }

                g = filter[0];
                val = intermediate[x + y * width];

                out += (val * g);

                // Testing last and fist diff
                val_2 = buffer[span * it.get_local_range(1) + it.get_local_id(1)];
                out_3 += (val_2 * g);
                syclexp::printf("SELF AKA 0 =%.10f -- Final out out =%.10f FILTER=%.20f --> val_2=%.10f - out_2=%.10f "
                                "- out_3=%.10f -- added_to_out_3= %.10f\n\n\n",
                                val,
                                out,
                                g,
                                val_2,
                                out_2,
                                out_3,
                                (val_2 * g));

#endif
            }

            val = buffer[span * it.get_local_range(1) + it.get_local_id(1)];
            out += (val * filter[0]);

            float out_sep = val * filter[0];

            if(it.get_global_linear_id() == 0)
            {
                syclexp::printf("Final out when adding self last ==> %.10f\n", out);
            }

            //
            data_array[0][write_y * dst_w + write_x] = out; // Store synchronously add asycn option for test later
        }

        // int i_min = 0
        // 0 is the min that we can go to

        // Move intermediate pointer to next row above so that we can just subtract one dst_w of the pointer each
        // iteration to load correct positon ( span + 1) * dst_w we need to subtract to do that

        std::optional<sycl::device_event> next_row_event; // Avoids using deleted default constructor of device_event
        int next_row_fetch = write_y - (span + 1);        // Set it to top row aswell so we can just decrement the value

        // if(!next_row_event.has_value())
        // {
        //     if(it.get_global_linear_id() == 0)
        //     {
        //         syclexp::printf("Optional does not have a value\n");
        //     }
        // }
        // next_row_event->wait(); // WAit on a non existing event

        // if(write_y - (span + 1) >= 0)

        if(next_row_fetch >= 0)
        {
            // If greater than zero we can load next row
            intermediate_ptr -= (span + 1) * dst_w;
            next_row_event = group.async_work_group_copy(
              buffer_ptr + ((span << 1) + 1) * it.get_local_range(1), intermediate_ptr, row_width);
        }
        else
        {
            // We need to copy from local memory to the location to avoid if's later
            buffer[((span << 1) + 1) * it.get_local_range(1) + it.get_local_id(1)] = buffer[it.get_local_id(1)];
        }

        // Now we have intermediate_ptr at top and we can load that into the free row

        // int i_max = dst_h - write_y - 1;
        // if (write_y span + 1

#if true
        write_y--;              // Decrement write_y as we have done the first iteration
        int free = (span << 1); // As we have already done it when free is at (span<<1) +  1 location (final pos)
        int end_pos = (it.get_global_id(0) * sg_region.height);
        // for(; write_y >= end_pos; write_y--)
        while(true)
        {
            // Move the other way for potential better cache

            // free row starts at bottom of window (with data in row below as that is how it's left after initial
            // work pre loop)
            for(int free = (span << 1); free >= 0; free--)
            // can stop and start at these position as then next loop it will be wrong
            {
                // Inner loop that moves the prefetch location

                int row_pos = free - (span + 1);
                int self_loop_size;
                if(row_pos >= 0)
                {
                    val = buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)];
                    out = val * filter[0]; // Reset by setting

                    // compute size of loop around self and implicitly loop size of around free
                    self_loop_size = row_pos; // Dist to top
                }
                else
                {
                    // wrap around
                    row_pos += (span << 1) + 2;
                    val = buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)];
                    out = val * filter[0]; // Reset by setting

                    // compute size of loop around self and implicitly loop size of around free
                    self_loop_size = ((span << 1) + 1) - row_pos; // Dist to botom
                }
                // TODO: Folow outline below and check that the whole logic works with wraparound and such

                // Now we have done self (the row we are writing to where the weight is only used once)
                // now we need to do the span on each side

                //

                // Do the loop around row_pos

                // by copying for clamping we don't need any boundary checks for these two loops
                int offset = 1;
                for(; offset < self_loop_size; ++offset)
                {
                    // Load value around self
                    val_above = buffer[(row_pos - offset) * it.get_local_range(1) + it.get_local_id(1)];
                    val_below = buffer[(row_pos + offset) * it.get_local_range(1) + it.get_local_id(1)];

                    // out += (val_above * filter[offset]);
                    // out += (val_below * filter[offset]);
                    out += ((val_above + val_below) * filter[offset]);
                    // Using one out is same as two atleast from sample test (with respect to precision)
                }
                // Moving final iteration out of loop so that we can wait on prev row load first
                // Final row could either be final iteration for self loop or final iteration for free loop so
                // waiting here ensures it's loaded and avoids if inside of loop. Should also in most cases be late
                // enough that it won't hamper performance too much (Could do if instead inside of loop but I think
                // that's worse) // In some cases it will be bad when self is at top of window all is around free
                // hance it's done straight away
                if(next_row_event.has_value())
                {
                    next_row_event->wait();
                }

                // Load next_row_value here
                write_y--; // Decrement here (We need to add one when writing result) This decrement is a bit early
                           // as it's for next already while we have not written current yet

                if(write_y >= end_pos)
                {
                    // Load in next row either async or copy from local to work as clamp
                    if((write_y - (span + 1)) >= 0)
                    {
                        intermediate_ptr -= dst_w; // Move to row above

                        next_row_event = group.async_work_group_copy(
                          buffer_ptr + free * it.get_local_range(1), intermediate_ptr, row_width);
                    }
                    else
                    {
                        // clamp to edge by copying data from prev row(row below)
                        // If multiple needed it works as a chain and as intended
                        int row_below = free + 1;
                        if(row_below <= (span << 1) + 1) // Final row of widow
                        {
                            buffer[free * it.get_local_range(1) + it.get_local_id(1)] =
                              buffer[row_below * it.get_local_range(1) + it.get_local_id(1)];
                        }
                        else
                        {
                            // Free is row at bottom of window so we need to load from top as that is whats below it
                            // when we wrap around
                            buffer[free * it.get_local_range(1) + it.get_local_id(1)] = buffer[it.get_local_id(1)];
                        }
                    }
                }

                // Offset should be equal to self_loop_size now as that is the value when it terminated loop above
                if(self_loop_size != 0)
                {
                    val_above = buffer[(row_pos - self_loop_size) * it.get_local_range(1) + it.get_local_id(1)];
                    val_below = buffer[(row_pos + self_loop_size) * it.get_local_range(1) + it.get_local_id(1)];

                    // out += (val_above * filter[offset]);
                    // out += (val_below * filter[offset]);
                    out += ((val_above + val_below) * filter[offset]);

                    // Increment offset here so that we are at correct offset when this runs and when we only loop
                    // around free
                    offset++;
                }

                for(int i = span - self_loop_size; i > 0; --i)
                {
                    // loop around free row
                    val_above = buffer[(free - i) * it.get_local_range(1) + it.get_local_id(1)];
                    val_below = buffer[(free + i) * it.get_local_range(1) + it.get_local_id(1)];
                    // out += (val_above * filter[offset]);
                    // out += (val_below * filter[offset]);
                    out += ((val_above + val_below) * filter[offset]);

                    offset++; // only increment at end so that when ours is at top and bottom of window we use correct
                              // filter
                }

                if(live)
                {
                    // We need to pluss one to write_y as we did decrement it a bit early
                    // So that we are writing to correct position
                    data_array[0][(write_y + 1) * dst_w + write_x] = out; // Synchronous setting
                }

                if(write_y < end_pos) // Next check is here we currently at write_y + 1
                {
                    break; // We are done // Outer loop will also exit at this point
                }
            }
            if(write_y < end_pos)
            {
                break; // Break out of otuer loop
            }
            // Otherwise reset free and continue

            // Set free to final pos as we have not had one iteration outside of the loop now
            int free = (span << 1) + 1;
        }
#endif
#if DO_HORIZ
        vert_sync_for_horiz(sg_region.wg_sync_state, it, 2);
#else
        vert_sync_for_horiz(
          sg_region.wg_sync_state, it, 1); // Set to one as we have not done horiz and hence can't get to that
#endif

        // Then we do DoG -- Or do DoG as we are storing the next level as we have the value in register and
        // only need to load one value (prev_level) and then we can store DoG as we go giving another
        // justification for doing it the persistent way and perhaps faster :D
#endif
    }
};

} // namespace normalizedSource

#if USE_PERSISTENT
// bool Pyramid::build_octave_one_wave_input(const Config& conf,
sycl::event Pyramid::build_octave_one_wave_input(const Config& conf,
                                                 ImageBase* base,
                                                 sycl::event d_gauss_write,
                                                 sycl::event img_write)
{
    Octave& oct_obj = _octaves[0];
    persistent_pyramid_octave_config& sg_region = oct_obj._sg_region;

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();
    // persistent_pyramid_octave_config sg_region = compute_persistent_sg_region_block(width, height);

    fprintf(stderr,
            "Local(%zu, %zu) -- global (%zu, %zu) -- sg_region --  width = %d - height = %d -- x_remainder = %d --"
            " y_remainder = % d\n\n ",
            sg_region.local[0],
            sg_region.local[1],
            sg_region.global[0],
            sg_region.global[1],
            sg_region.sg_block.width,
            sg_region.sg_block.height,
            sg_region.x_remainder,
            sg_region.y_remainder);

    // if(!sg_region.use_persistent_block)
    //     return false; // Could not use persistent block

#if true
    for(auto& plat : sycl::platform::get_platforms())
    {
        std::cout << "CUDA‐SYCL platform name: " << plat.get_info<sycl::info::platform::name>() << "\n"
                  << "Reported version:     " << plat.get_info<sycl::info::platform::version>() << "\n";
    }

#if SYCL_EXT_ONEAPI_ROOT_GROUP == 1
    printf("ROOT GPOUP SUPPPOERTED\n");
#else
    printf("ROOT GROUP NOT SUPPORTED\n");
#endif

#endif

    // Just for test
    if(!sg_region.use_persistent_block)
        return sycl::event(); // Could not use persistent block

    // Use persistent block

    // Lauch kernel that starts from input image -- quite likely to be compatible with persistent_block

    // sycl::event dependency = prev_oct_obj.getLevelCompleteEvent(_levels - PREV_LEVEL);

    // This ^ is the event we need to write to ensure interoperability between block and non block versions running.
    // So for from prev version this would also need to be our dependency (not sure if that is good enough not sure
    // this would work in case of other kernels being in flight...)

#if USE_ROOT_GROUP
    auto props = syclexp::properties{syclexp::use_root_sync}; // Does not seem to be supported by cuda backend for dpc++
#endif
    if constexpr(USE_BINDLESS_INPUT && sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d>())
    {
        // auto kernel_id = sycl::get_kernel_id<horiz_if>();
        // auto kernel_bundle =
        // sycl::get_kernel_bundle<sycl::bundle_state::executable>(_device_queue.get_context()); auto kernel =
        // kernel_bundle.get_kernel(kernel_id);
        //
        // auto compiled_num_sg = kernel.template
        // get_info<sycl::info::kernel_device_specific::compile_num_sub_groups>(
        //   _device_queue.get_device());
        // auto max_num_sg =
        //   kernel.template
        //   get_info<sycl::info::kernel_device_specific::max_num_sub_groups>(_device_queue.get_device());
        // auto prefered_wg_multiple =
        //   kernel.template get_info<sycl::info::kernel_device_specific::preferred_work_group_size_multiple>(
        //     _device_queue.get_device());

        // Bindless version
        const float shift = 0.5f * sycl::pow(2.0f, conf.getUpscaleFactor());

        const bool col = sg_region.x_remainder != 0;
        const bool row = sg_region.y_remainder != 0;

        // auto device = _device_queue.get_device();
        //
        // // Get local memory size in bytes
        // size_t local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        //
        const int buffer_size = (sg_region.local[1] + (Pyramid::largest_span << 2)) * (sg_region.local[0] << 2);
        std::printf("Buffer_size = %d -- sg_region.local_mem_size = %d\n", buffer_size, sg_region.local_mem_size);

        printf("\tSG_REGION_HEIGHT = %d\n\n", sg_region.sg_block.height);
        // const int vert_buffer_size =
        //   ((sg_region.local[1] * 13) * sg_region.local[0]); // might be better to store the 13 values in
        //   registers
        // might be problematic if register pressure get's too high
        // printf("Local(%zu, %zu) -- global(%zu, %zu) --- Local mem size = %zu -- largest span = %d\n",
        //        sg_region.local[0],
        //        sg_region.local[1],
        //        sg_region.global[0],
        //        sg_region.global[1],
        //        local_mem_size,
        //        largest_span);
        // printf("THIS IS SHIFT = %f -- widht=%d -- height=%d  -- col = %d -- row = %d -- Buffer_size = %d -- "
        //        "vert_buffer_size = %d",
        //        shift,
        //        width,
        //        height,
        //        col,
        //        row,
        //        buffer_size,
        //        vert_buffer_size);

        // sg_region.local_mem_size
        if(col && row)
        {
            // printf("We doing col and row whop whop\n");
            // sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // TODO: Need to figure out what type of buffer I need for vert and assign the largest one to use

                // auto buffer = sycl::local_accessor<float, 1>(buffer_size, cgh);
                auto buffer = sycl::local_accessor<float, 1>(sg_region.local_mem_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
#if USE_ROOT_GROUP
                                 props,
#endif
                                 normalizedSource::BuildOctave<true, true>(base->getInputImage(),
                                                                           oct_obj.getDataArray(),
                                                                           oct_obj.getDogArray(),
                                                                           oct_obj.getIntermediate(),
                                                                           _d_gauss,
                                                                           buffer,
                                                                           sg_region.sg_block,
                                                                           width,
                                                                           height,
                                                                           shift,
                                                                           _levels));
            });
        }
        else if(col)
        {
            // sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // TODO: Need to figure out what type of buffer I need for vert and assign the largest one to use

                // auto buffer = sycl::local_accessor<float, 1>(buffer_size, cgh);
                auto buffer = sycl::local_accessor<float, 1>(sg_region.local_mem_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
#if USE_ROOT_GROUP
                                 props,
#endif
                                 normalizedSource::BuildOctave<true, false>(base->getInputImage(),
                                                                            oct_obj.getDataArray(),
                                                                            oct_obj.getDogArray(),
                                                                            oct_obj.getIntermediate(),
                                                                            _d_gauss,
                                                                            buffer,
                                                                            sg_region.sg_block,
                                                                            width,
                                                                            height,
                                                                            shift,
                                                                            _levels));
            });
        }
        else if(row)
        {
            // sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // TODO: Need to figure out what type of buffer I need for vert and assign the largest one to use

                // auto buffer = sycl::local_accessor<float, 1>(buffer_size, cgh);
                auto buffer = sycl::local_accessor<float, 1>(sg_region.local_mem_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
#if USE_ROOT_GROUP
                                 props,
#endif
                                 normalizedSource::BuildOctave<false, true>(base->getInputImage(),
                                                                            oct_obj.getDataArray(),
                                                                            oct_obj.getDogArray(),
                                                                            oct_obj.getIntermediate(),
                                                                            _d_gauss,
                                                                            buffer,
                                                                            sg_region.sg_block,
                                                                            width,
                                                                            height,
                                                                            shift,
                                                                            _levels));
            });
        }
        else
        {
            // sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // TODO: Need to figure out what type of buffer I need for vert and assign the largest one to use

                // auto buffer = sycl::local_accessor<float, 1>(buffer_size, cgh);
                auto buffer = sycl::local_accessor<float, 1>(sg_region.local_mem_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
#if USE_ROOT_GROUP
                                 props,
#endif
                                 normalizedSource::BuildOctave<false, false>(base->getInputImage(),
                                                                             oct_obj.getDataArray(),
                                                                             oct_obj.getDogArray(),
                                                                             oct_obj.getIntermediate(),
                                                                             _d_gauss,
                                                                             buffer,
                                                                             sg_region.sg_block,
                                                                             width,
                                                                             height,
                                                                             shift,
                                                                             _levels));
            });
        }

        // _device_queue.wait(); // For testing
    }
    else
    {
        // Normal inpute image not bindless

        // sycl::event e = _device_queue.parallel_for(
        //   sycl::nd_range{global, local},
        //   {d_gauss_write, img_write},
        //   absoluteSource::Horiz<0, true>(base->getInputFloat(), oct_obj.getIntermediate(), _d_gauss, width,
        //   height, 0));
    }

    return sycl::event();
}
#endif // if USE_PERSISTENT

} // namespace popsift

#if false
        // Position to write to (image that has the size of scale up)
        const int write_x = it.get_global_id(1);
        // const int write_y = it.get_global_id(0) * dst_w;
        const int write_y = it.get_group(0);
        // Cant use it.get_global_range(1) inplace of dst_w due to if if_required width !=
        it.get_global_range(1) and
          // hence positions would be off could be used in else case but not sure if it matters much (probs not)

          if constexpr(if_required)
        {
            // Destination width was not perfectly divisible with it.get_local_range(1)
            if(write_x >= dst_w)
                return;
        }

        const float* filter = &d_gauss->dd.filter[0];
        const int span = d_gauss->dd.span[0];
        const float read_x = (write_x + shift) / dst_w;
        const float read_y = (write_y + shift) / dst_h;

        // Could pass dimensions as a int2 and do vector wise
        // const sycl::float2 read_pos = sycl::float2{(write_x + shift) / dst_w, (write_y + shift) /
        dst_h
    };

    float out = 0.0f;

    // Look into sycl mad or fma (multiply-and-add instruction done in one clock cycle)
    // is probably done by the compiler anyways though

#pragma unroll
    for(int offset = span; offset > 1; offset--)
    {
        const float g = filter[offset];
        const float offrel = float(offset) / dst_w; // relative offset
        const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
        const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
        out += ((v1 + v2) * g);
    }

    const float& g = filter[0];
    const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    out += (v3 * g);

    dst_data[write_x + write_y * dst_w] = out * 255.0f;
#endif
