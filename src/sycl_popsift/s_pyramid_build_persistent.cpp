#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/persistent_config_macros.h" // If we are using root group or handcrafted wg syncrinozation
#include "sycl_popsift/persistent_configuration.hpp"
#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/sift_constants.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

#include <iterator>
#include <optional>

#define USE_SHARED_MEM_FOR_INPUT 0

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
                                        float read_y)
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
        const float offrel = float(offset) / dst_w; // relative offset
        const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
        const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
        out += ((v1 + v2) * g);
    }

    const float& g = filter[0];
    const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    out += (v3 * g);

    intermediate[write_x + write_y * dst_w] = out * 255.0f;
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

        const float v1 = buffer[base_pos - offset];
        const float v2 = buffer[base_pos + offset];
        out += ((v1 + v2) * g);
    }

    out += (buffer[base_pos] * filter[0]);

    intermediate[write_x + write_y * dst_w] = out * 255.0f;
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
                         // sycl::memory_order_relaxed,
                         sycl::memory_order_seq_cst,
                         sycl::memory_scope_device,
                         sycl::access::address_space::global_space>(wg_sync_state[group_linear_pos])++;

        // Active wait-- spin lock
        if(group_pos == 0)
        {
            // left border -- only depends on right

            sycl::atomic_ref<int,
                             // sycl::memory_order_relaxed,
                             sycl::memory_order_seq_cst,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              right(wg_sync_state[group_linear_pos + 1]);

            while(right < wait_on_state) {}
        }
        else if(group_pos == group_final_index)
        {
            // right most border -- only depends on left

            sycl::atomic_ref<int,
                             // sycl::memory_order_relaxed,
                             sycl::memory_order_seq_cst,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              left(wg_sync_state[group_linear_pos - 1]);
            while(left < wait_on_state) {}
        }
        else
        {
            // Normal in the middle  -- depends on left and right
            sycl::atomic_ref<int,
                             // sycl::memory_order_relaxed,
                             sycl::memory_order_seq_cst,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              left(wg_sync_state[group_linear_pos - 1]);

            sycl::atomic_ref<int,
                             // sycl::memory_order_relaxed,
                             sycl::memory_order_seq_cst,
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
static inline void full_sync(int* wg_sync_state, sycl::nd_item<2>& it, int level)
{
#if USE_ROOT_GROUP
    auto root = it.ext_oneapi_get_root_group(); // Root group all work_items running kernel
    sycl::group_barrier(root);
#else

    sycl::group group = it.get_group();
    sycl::group_barrier(group); // Ensure all have done horiz

    if(it.get_local_linear_id() == 0) // only one per work_group
    {
        // Need to be a 4 byte wide data type like int or unsigned int for this to work...
        sycl::atomic_ref<int,
                         // sycl::memory_order_relaxed,
                         sycl::memory_order_seq_cst,
                         sycl::memory_scope_device,
                         sycl::access::address_space::global_space>(wg_sync_state[level])++;
        // All use zero so that we count everyone and once all have reached we continue

        // Active wait-- spin lock

        int num_work_groups = it.get_group_range(0) * it.get_group_range(1);

        sycl::atomic_ref<int,
                         // sycl::memory_order_relaxed,
                         sycl::memory_order_seq_cst,
                         sycl::memory_scope_device,
                         sycl::access::address_space::global_space>
          state(wg_sync_state[level]);

        int copy_state = state;

        // Active wait -- spin lock (waits for everyone to have done horiz before moving on (SLOW))
        while(state < num_work_groups) {}
    }
    sycl::group_barrier(group); // Wait for wg leader to finish spin lock ensuring dependencies are done
#endif
}
// synchronizes horiz execution so that all data needed to do vert is available and correct
static inline void horiz_sync_for_vert(int* wg_sync_state, sycl::nd_item<2>& it, int wait_on_state)
{
#if USE_ROOT_GROUP
    auto root = it.ext_oneapi_get_root_group(); // Root group all work_items running kernel
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
                         // sycl::memory_order_relaxed,
                         sycl::memory_order_seq_cst,
                         sycl::memory_scope_device,
                         sycl::access::address_space::global_space>(wg_sync_state[it.get_group_linear_id()])++;

        // Active wait-- spin lock
        if(group_pos_0 == 0)
        {
            // top border -- only depends on wg below

            sycl::atomic_ref<int,
                             // sycl::memory_order_relaxed,
                             sycl::memory_order_seq_cst,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              below(wg_sync_state[(group_pos_0 + 1) * group_range_1 + group_pos_1]);

            while(below < wait_on_state) {}
        }
        else if(group_pos_0 == group_range_0 - 1)
        {
            // right most border -- only depends on left

            sycl::atomic_ref<int,
                             // sycl::memory_order_relaxed,
                             sycl::memory_order_seq_cst,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              above(wg_sync_state[(group_pos_0 - 1) * group_range_1 + group_pos_1]);
            while(above < wait_on_state) {}
        }
        else
        {
            // Normal in the middle  -- depends on left and right
            sycl::atomic_ref<int,
                             // sycl::memory_order_relaxed,
                             sycl::memory_order_seq_cst,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              above(wg_sync_state[(group_pos_0 - 1) * group_range_1 + group_pos_1]);

            sycl::atomic_ref<int,
                             // sycl::memory_order_relaxed,
                             sycl::memory_order_seq_cst,
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

// One by one results in two more registers being used...
#define COMPUTE_ONE_BY_ONE 0 // If true we do out += (above * g); out += (below * g); else out += ((above + below) * g);

template<bool REMAINDER_COL, bool REMAINDER_ROW>
static inline void horiz_persistent_bindless(syclexp::sampled_image_handle src,
                                             float* intermediate,
                                             // sycl::local_accessor<float, 1> buffer,
                                             // popsift::GaussInfo* d_gauss,
                                             const float* filter,
                                             const int span,
                                             const float shift,
                                             const int sg_region_height,
                                             const int dst_w,
                                             const int dst_h,
                                             const int write_x,
                                             int& write_y,
                                             // int level,
                                             sycl::nd_item<2>& it)
// (float* intermediate,
//                                     sycl::local_accessor<float, 1> buffer,
//                                     const float* filter,
//                                     const int span,
//                                     const int dst_w,
//                                     const int write_x,
//                                     int write_y,
//                                     int base_pos)
{
    // #if REMAINDER_COL
    //     if(write_x >= dst_w)
    //         return;
    // #endif
    //
    //     float out = 0.0f;
    //
    // #pragma unroll
    //     for(int offset = span; offset > 0; offset--)
    //     {
    //         const float g = filter[offset];
    //
    //         const float v1 = buffer[base_pos - offset];
    //         const float v2 = buffer[base_pos + offset];
    //         out += ((v1 + v2) * g);
    //     }
    //
    //     out += (buffer[base_pos] * filter[0]);
    //
    //     intermediate[write_x + write_y * dst_w] = out * 255.0f;

    // #####################################################
    // End of simple vert function
    // #####################################################

    // #if INITIAL // ALWAYS IS
    //     const float* filter_input = &d_gauss->dd.filter[0];
    //     const int span_input = d_gauss->dd.span[0];
    // #else
    //
    //     const float* filter = &d_gauss->inc.filter[level * GAUSS_ALIGN];
    //     const int span = d_gauss->inc.span[level];
    //
    // #if MINIMAL_WINDOW
    //     const int span_width = d_gauss->inc.span[0] - 1;
    // #else
    //     const int span_width = d_gauss->inc.span[0];
    // #endif

    // #endif

    const float read_x = (write_x + shift) / dst_w;
    float read_y = (write_y + shift) / dst_h;

#if USE_SHARED_MEM_FOR_INPUT
    // Not sure if there is a point of using this for input level -- As we can't async load
    const int base_pos = (it.get_local_range(1) + (span << 1)) * (it.get_local_id(0) << 1) + it.get_local_id(1) + span;

    // Second buffer row (there are two per row in the work-group)
    const int base_pos_2 =
      (it.get_local_range(1) + (span << 1)) * ((it.get_local_id(0) << 1) + 1) + it.get_local_id(1) + span;

    // const int rel_span = ((1 / dst_w) * span); // Relative span value used for offset
    const float rel_span = float(span) / dst_w; // Relative span value used for offset
#endif

    // for(int i = 0; i < sg_region.height; i++)

    // const float read_y_increment = 1.0f / dst_h; // Does not result in the same as recompute due to
    // accumulation of floating point error

    int loop_end = write_y + sg_region_height;

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

        if(it.get_local_id(1) < span)
        {
            // load left side (lenght of span)
            buffer[base_pos - span] = syclexp::sample_image<float>(src, sycl::float2{read_x - rel_span, read_y});
        }
        else if(it.get_local_id(1) >= (it.get_local_range(1) - span))
        {
            buffer[base_pos + span] = syclexp::sample_image<float>(src, sycl::float2{read_x + rel_span, read_y});
        }

        // Here would be good to do async load of next row but does not seem to be possible to do with bindless
        // images But for remaining parts it will be not sure if we should use local mem for this part however
        sycl::group_barrier(it.get_group()); // Ensure all is loaded before we do horiz

        horiz_local_mem<REMAINDER_COL>(intermediate, buffer, filter, span, dst_w, write_x, write_y, base_pos);

#else
        horiz_bindless_input<REMAINDER_COL>(intermediate, src, filter, span, dst_w, write_x, write_y, read_x, read_y);
#endif
        // Second row buffer in use: Same as above otherwise

        write_y++;
        if(write_y >= loop_end)
            break;

        // read_y += read_y_increment; // Floating point error accumulation hence not using
        read_y = (write_y + shift) / dst_h;

#if USE_SHARED_MEM_FOR_INPUT
        buffer[base_pos_2] = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});

        if(it.get_local_id(1) < span)
        {
            buffer[base_pos_2 - span] = syclexp::sample_image<float>(src, sycl::float2{read_x - rel_span, read_y});
        }
        else if(it.get_local_id(1) >= (it.get_local_range(1) - span))
        {
            buffer[base_pos_2 + span] = syclexp::sample_image<float>(src, sycl::float2{read_x + rel_span, read_y});
        }

        sycl::group_barrier(it.get_group());

        horiz_local_mem<REMAINDER_COL>(intermediate, buffer, filter, span, dst_w, write_x, write_y, base_pos_2);
#else
        horiz_bindless_input<REMAINDER_COL>(intermediate, src, filter, span, dst_w, write_x, write_y, read_x, read_y);
#endif
    }
    // At end of this loop write_y will be equal to loop_end or smaller but it wil always be one too large hence
    // need to subtract one
    write_y--;

    // DONE WITH INITIAL HORIZ

    //     // Synchronize and then do horiz
    // #define USE_FULL_SYNC 0
    // #if USE_FULL_SYNC
    //     // Ensures all work groups have completed horiz before moving on to vert
    //     full_sync(sg_region.wg_sync_state, it);
    //
    //     // if(it.get_local_linear_id() == 0)
    //     // {
    //     //     syclexp::printf("WG_ID %d --> Num_wg = %d\n",
    //     //                     static_cast<int>(it.get_group_linear_id()),
    //     //                     static_cast<int>(it.get_group_range(0) * it.get_group_range(1)));
    //     // }
    //     //
    //     // horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);
    // #else
    //     // Only necessary negbours (top and bottom) are waited on
    //     horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);
    // #endif
    //
    //     // #if false
    //     // Start doing Vert then later we do horiz on data_array so not using sampled image then we can use async
    //     // load of next row Do vert for this one then make loop over the levels for the rest with horiz from prev
    //     // and vert from intermediate
    //
    //     // Vert
    //     sycl::group_barrier(it.get_group());
}

template<bool REMAINDER_ROW>
static inline void horiz_persistent(float* intermediate,
                                    float** data_array,
                                    // sycl::local_accessor<float, 1> buffer,
                                    // popsift::GaussInfo* d_gauss,
                                    float* filter,
                                    const int span_width,
                                    const int sg_region_height,
                                    const int dst_w,
                                    const int dst_h,
                                    const int write_x,
                                    int& write_y,
                                    int prev_lvl)
// ,sycl::nd_item<2>& it)
{
    int loop_end = write_y + sg_region_height;

    if constexpr(REMAINDER_ROW)
    {
        if(loop_end >= dst_h)
            loop_end = dst_h; // Limit to last pixel
    }

    int self_pos = write_y * dst_w + write_x;

    if(write_x < dst_w) // to ensure no out of bounds reads and writes
    {
        for(; write_y < loop_end; ++write_y) // Modifies write_y want that later
        {
            float out = 0.0f;
            int idx;
            for(int span = span_width; span > 0; --span)
            {
                // int pos_left = self_pos - span;
                idx = write_x - span;
                float val_left =
                  idx >= 0 ? data_array[prev_lvl][self_pos - span] : data_array[prev_lvl][write_y * dst_w];
#if COMPUTE_ONE_BY_ONE
                out += val_left * filter[span];
#endif
                // int pos_right = self_pos + span;

                idx = write_x + span;
                float val_right =
                  idx < dst_w ? data_array[prev_lvl][self_pos + span] : data_array[prev_lvl][(write_y + 1) * dst_w - 1];
#if COMPUTE_ONE_BY_ONE
                out += val_right * filter[span];
#else
                // out += (val_above + val_below) * d_gauss->inc.filter[span];
                out += (val_left + val_right) * filter[span];
#endif
            }

            out += data_array[prev_lvl][self_pos] * filter[0];

            intermediate[self_pos] = out;

            self_pos += dst_w; // Move down to next row
        }
    }
}

// template<bool LAST_LVL, bool LVL_ZERO>

// MBY TEST ASYNC WRITE OF BOTH DOG AND DATA

template<bool DO_DOG, bool FINAL_LVL>
static inline void vert_persistent(float** data_array,
                                   float** dog_array,
                                   float* intermediate,
                                   // popsift::GaussInfo* d_gauss,
                                   float* filter,
                                   const int span_width,
                                   const int sg_region_height,
                                   const int dst_w,
                                   const int dst_h,
                                   const int write_x,
                                   int& write_y,
                                   int level,
                                   sycl::nd_item<2>& it)
{
    // #if MINIMAL_WINDOW
    //     const int span_width = d_gauss->inc.span[level] - 1;
    // #else
    //     const int span_width = d_gauss->inc.span[level];
    // #endif

    int end_pos = (it.get_global_id(0) * sg_region_height);
    const int pos_upper_limit = dst_w * dst_h; // First pixel outside of image bounds
    int self_pos = write_y * dst_w + write_x;

    if(write_x < dst_w)
    {
        for(; write_y >= end_pos; --write_y)
        {
            float out;
            int offset = span_width * dst_w;
            out = 0.0f;
            for(int span = span_width; span > 0; --span)
            {
                int pos_above = self_pos - offset;
                float val_above = pos_above >= 0 ? intermediate[pos_above] : intermediate[write_x];
#if COMPUTE_ONE_BY_ONE
                // out += val_above * d_gauss->inc.filter[span];
                out += val_above * filter[span];
#endif
                int pos_below = self_pos + offset;
                float val_below =
                  pos_below < pos_upper_limit ? intermediate[pos_below] : intermediate[(dst_h - 1) * dst_w + write_x];
#if COMPUTE_ONE_BY_ONE
                // out += val_below * d_gauss->inc.filter[span];
                out += val_below * filter[span];
#else
                // out += (val_above + val_below) * d_gauss->inc.filter[span];
                out += (val_above + val_below) * filter[span];
#endif

                offset -= dst_w;
            }
            // out += intermediate[self_pos] * d_gauss->inc.filter[0]; // Always safe
            out += intermediate[self_pos] * filter[0]; // Always safe

            // BUG: Not doing this for final level results in the results to be wrong... (FIX) Should not be needed
            // this is due to currently we are not using the DoG that we store here and rely on DoG normal

            // if constexpr(!FINAL_LVL)
            // {
            data_array[level][self_pos] = out; // guarded by outer if
            // }

            if constexpr(DO_DOG)
            {
                float prev_val = data_array[level - 1][self_pos];
                // if(it.get_global_linear_id() == 0)
                // {
                //     syclexp::printf("Running boys prev = %f -- cur = %f ", prev_val, out);
                // }
                dog_array[level - 1][self_pos] = out - prev_val;
            }
            self_pos -= dst_w;
        }
    }
}

// template<bool REMAINDER_COL, bool REMAINDER_ROW>
class BuildOctaveSimple
{
  private:
    syclexp::sampled_image_handle src;
    float** data_array; // Need to be array of all dst data
    float** dog_array;
    float* intermediate;
    popsift::GaussInfo* d_gauss;
    // sycl::local_accessor<float, 1> buffer;
    const sg_region_blocks sg_region;
    const int dst_w;
    const int dst_h;
    const float shift;
    const int levels;

  public:
    BuildOctaveSimple(syclexp::sampled_image_handle src,
                      float** data_array,
                      float** dog_array,
                      float* intermediate,
                      popsift::GaussInfo* d_gauss,
                      // sycl::local_accessor<float, 1> buffer,
                      const sg_region_blocks sg_region, // THINK WE ONLY NEED THE HEIGHT
                      const int dst_w,
                      const int dst_h,
                      const float shift,
                      const int levels)

      : src(src)
      , data_array(data_array)
      , dog_array(dog_array)
      , intermediate(intermediate)
      , d_gauss(d_gauss)
      // , buffer(buffer)
      , sg_region(sg_region)
      , dst_w(dst_w)
      , dst_h(dst_h)
      , shift(shift)
      , levels(levels) {};

    inline void operator()(sycl::nd_item<2> it) const
    {
        const int write_x = it.get_global_id(1);
        int write_y = it.get_global_id(0) * sg_region.height; // Changes in normal block

        const float* filter_input = &d_gauss->dd.filter[0];

#if MINIMAL_WINDOW
        const int span_input = d_gauss->inc.span[0] - 1;
#else
        const int span_input = d_gauss->inc.span[0];
#endif

        // horiz_persistent_bindless<REMAINDER_COL, REMAINDER_ROW>(
        int sim_write_y = sycl::min(write_y + sg_region.height - 1, dst_h - 1); // Simulating horiz run with it's end

        horiz_persistent_bindless<true, true>(
          src, intermediate, filter_input, span_input, shift, sg_region.height, dst_w, dst_h, write_x, write_y, it);

        // Simulate that we have done horiz before vert
        // write_y = sycl::min(write_y + sg_region.height - 1, dst_h - 1); // Simulating horiz run with it's end

        // horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);

        full_sync(sg_region.wg_sync_state, it, 0);

#if MINIMAL_WINDOW
        const int span_initial = d_gauss->inc.span[0] - 1;
#else
        const int span_initial = d_gauss->inc.span[0];
#endif
        float* filter = &d_gauss->inc.filter[0];
        // vert_persistent<false, false>(
        //   data_array, dog_array, intermediate, d_gauss, sg_region.height, dst_w, dst_h, write_x, write_y, 0, it);

        vert_persistent<false, false>(data_array,
                                      dog_array,
                                      intermediate,
                                      filter,
                                      span_initial,
                                      sg_region.height,
                                      dst_w,
                                      dst_h,
                                      write_x,
                                      write_y,
                                      0,
                                      it);

        full_sync(sg_region.wg_sync_state, it, 1);

        for(int lvl = 1; lvl < (levels - 1); ++lvl) // Stop before final level
        // for(int lvl = 1; lvl < levels; ++lvl)
        {
            filter += GAUSS_ALIGN; // Move to next level (same as level * GAUSS_ALIGN)
            horiz_persistent<true>(intermediate,
                                   data_array,
                                   filter,
#if MINIMAL_WINDOW
                                   d_gauss->inc.span[lvl] - 1,
#else
                                   d_gauss->inc.span[lvl],
#endif
                                   sg_region.height,
                                   dst_w,
                                   dst_h,
                                   write_x,
                                   write_y,
                                   lvl - 1);
            full_sync(sg_region.wg_sync_state, it, lvl << 1);

            vert_persistent<true, false>(data_array,
                                         dog_array,
                                         intermediate,
                                         filter,
#if MINIMAL_WINDOW
                                         d_gauss->inc.span[lvl] - 1,
#else
                                         d_gauss->inc.span[lvl],
#endif
                                         sg_region.height,
                                         dst_w,
                                         dst_h,
                                         write_x,
                                         write_y,
                                         lvl,
                                         it);
            full_sync(sg_region.wg_sync_state, it, (lvl << 1) + 1);
        }

        // Do the final level

        filter += GAUSS_ALIGN; // Move to next level (same as level * GAUSS_ALIGN)

        horiz_persistent<true>(intermediate,
                               data_array,
                               filter,
#if MINIMAL_WINDOW
                               d_gauss->inc.span[levels - 1] - 1,
#else
                               d_gauss->inc.span[levels - 1],
#endif
                               sg_region.height,
                               dst_w,
                               dst_h,
                               write_x,
                               write_y,
                               levels - 2);

        full_sync(sg_region.wg_sync_state, it, (levels - 1) << 1);

        vert_persistent<true, true>(data_array,
                                    dog_array,
                                    intermediate,
                                    filter,
#if MINIMAL_WINDOW
                                    d_gauss->inc.span[levels - 1] - 1,
#else
                                    d_gauss->inc.span[levels - 1],
#endif
                                    sg_region.height,
                                    dst_w,
                                    dst_h,
                                    write_x,
                                    write_y,
                                    levels - 1,
                                    it);
    }
};

// Used for ImageBindless
// Only used on input image (initial)
// And only works for it due to  filter and span selection

// aspect::ext_oneapi_bindless_sampled_image_fetch_2d
// This aspect is required to use sampled image need to add a check for that earlier in selection
// template<bool if_required>

#define DO_HORIZ 0
#define DO_VERT 1

#define DEBUG 0

// Uses prefetch of 2 rows for vert
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
#define USE_FULL_SYNC 0
#if USE_FULL_SYNC
        // Ensures all work groups have completed horiz before moving on to vert
        full_sync(sg_region.wg_sync_state, it);

        // if(it.get_local_linear_id() == 0)
        // {
        //     syclexp::printf("WG_ID %d --> Num_wg = %d\n",
        //                     static_cast<int>(it.get_group_linear_id()),
        //                     static_cast<int>(it.get_group_range(0) * it.get_group_range(1)));
        // }
        //
        // horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);
#else
        // Only necessary negbours (top and bottom) are waited on
        horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);
#endif

        // #if false
        // Start doing Vert then later we do horiz on data_array so not using sampled image then we can use async
        // load of next row Do vert for this one then make loop over the levels for the rest with horiz from prev
        // and vert from intermediate

        // Vert
        sycl::group_barrier(it.get_group());

#endif
#if DO_VERT

#if !DO_HORIZ
        // So it is the same it would have been if we were doing horiz
        write_y = sycl::min(write_y + sg_region.height - 1, dst_h - 1);
#endif

        const int row_width = (it.get_group(1) == it.get_group_range(1) - 1)
                                ? it.get_local_range(1) - (it.get_global_range(1) - dst_w)
                                : it.get_local_range(1);

        const bool live = it.get_local_id(1) < row_width;

        int start_pos = write_y * dst_w + it.get_group(1) * it.get_local_range(1);
        // Bottom row position
        auto intermediate_ptr =
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
            intermediate + start_pos);

        // auto inter =
        //   sycl::make_ptr<sycl::access::address_space::global_space, sycl::access::decorated::yes>(intermediate);
        // auto intermeidate_ptr = inter.template get_multi_ptr<sycl::access::decorated::yes>();
        auto buffer_ptr = buffer.template get_multi_ptr<sycl::access::decorated::yes>();

        float* filter = &d_gauss->inc.filter[0]; // For level 0
        sycl::group group = it.get_group();

        int end_pos = (it.get_global_id(0) * sg_region.height);
        float out;

        // If not live it does everything besides write as it's comoputing non-sense
#if MINIMAL_WINDOW
        const int span_width = d_gauss->inc.span[0] - 1;
#else
        const int span_width = d_gauss->inc.span[0];
#endif

        // int span;
        // int span_bottom_treshold;
        // int prefetch_span;
#if !COMPUTE_ONE_BY_ONE
        float val_above, val_below;
#endif

        for(; write_y >= end_pos; --write_y)
        {
            int span = span_width; // reset span

            // Need to modify intermediate pointer
            // How many pixels we can go down before we are beyond image bounds
            int span_bottom_treshold = dst_h - 1 - write_y;

            // NOTE: Assumes that span cannot be less than 4
            // -> if that is not the case it will do unnecessary loads but won't cause it to compute wrong value or fail

            int prefetch_span = span_width; // reset
            // Store at rowIdx 2
            sycl::device_event above_1 =
              (prefetch_span <= write_y)
                ? group.async_work_group_copy(
                    buffer_ptr + (it.get_local_range(1) << 1), intermediate_ptr - (dst_w * prefetch_span), row_width)
                : group.async_work_group_copy(
                    buffer_ptr + (it.get_local_range(1) << 1), intermediate_ptr - (dst_w * write_y), row_width);

            // Store at rowIdx 3
            sycl::device_event below_1 =
              (prefetch_span <= span_bottom_treshold)
                ? group.async_work_group_copy(
                    buffer_ptr + it.get_local_range(1) * 3, intermediate_ptr + (dst_w * prefetch_span), row_width)
                : group.async_work_group_copy(buffer_ptr + it.get_local_range(1) * 3,
                                              intermediate_ptr + (dst_w * span_bottom_treshold),
                                              row_width);

            prefetch_span--; // Go next row inwards
            // second outermost row pair
            // Store at rowIdx 0
            sycl::device_event above_0 =
              (prefetch_span <= write_y)
                ? group.async_work_group_copy(buffer_ptr, intermediate_ptr - (dst_w * prefetch_span), row_width)
                : group.async_work_group_copy(buffer_ptr, intermediate_ptr - (dst_w * write_y), row_width);

            // Stor at rowIdx 1
            sycl::device_event below_0 =
              (prefetch_span <= span_bottom_treshold)
                ? group.async_work_group_copy(
                    buffer_ptr + it.get_local_range(1), intermediate_ptr + (dst_w * prefetch_span), row_width)
                : group.async_work_group_copy(
                    buffer_ptr + it.get_local_range(1), intermediate_ptr + (dst_w * span_bottom_treshold), row_width);

#if COMPUTE_ONE_BY_ONE
            above_1.wait();
            out = buffer[(it.get_local_range(1) << 1) + it.get_local_id(1)] * filter[span];

            below_1.wait();
            out += buffer[(it.get_local_range(1) * 3) + it.get_local_id(1)] * filter[span];
#else
            above_1.wait();
            val_above = buffer[(it.get_local_range(1) << 1) + it.get_local_id(1)];

            below_1.wait();
            val_below = buffer[(it.get_local_range(1) * 3) + it.get_local_id(1)];

            out = (val_above + val_below) * filter[span]; // reset by setting
#endif

            prefetch_span--;
            above_1 = (prefetch_span <= write_y)
                        ? group.async_work_group_copy(buffer_ptr + (it.get_local_range(1) << 1),
                                                      intermediate_ptr - (dst_w * prefetch_span),
                                                      row_width)
                        : group.async_work_group_copy(
                            buffer_ptr + (it.get_local_range(1) << 1), intermediate_ptr - (dst_w * write_y), row_width);

            below_1 = (prefetch_span <= span_bottom_treshold)
                        ? group.async_work_group_copy(buffer_ptr + it.get_local_range(1) * 3,
                                                      intermediate_ptr + (dst_w * prefetch_span),
                                                      row_width)
                        : group.async_work_group_copy(buffer_ptr + it.get_local_range(1) * 3,
                                                      intermediate_ptr + (dst_w * span_bottom_treshold),
                                                      row_width);
            // while(span > 0) // loop over current to compute the out value // Could replace this one with shared
            // memory
            while(true)
            {
                span--;
                if(span == 0) // Level of which we process
                {
                    // Do self -- always safe
                    above_0.wait();
                    out += buffer[it.get_local_id(1)] * filter[0];
                    break;
                }

                prefetch_span--;

#if COMPUTE_ONE_BY_ONE
                above_0.wait();
                out += buffer[it.get_local_id(1)] * filter[span];

                below_0.wait();
                out += buffer[it.get_local_range(1) + it.get_local_id(1)] * filter[span];
#else

                above_0.wait();
                val_above = buffer[it.get_local_id(1)];

                below_0.wait();
                val_below = buffer[it.get_local_range(1) + it.get_local_id(1)];
                out += (val_above + val_below) * filter[span];
#endif

                // Prefetch next row
                if(prefetch_span > 0)
                {
                    // prefetch normal
                    above_0 =
                      (prefetch_span <= write_y)
                        ? group.async_work_group_copy(buffer_ptr, intermediate_ptr - (dst_w * prefetch_span), row_width)
                        : group.async_work_group_copy(buffer_ptr, intermediate_ptr - (dst_w * write_y), row_width);

                    below_0 = (prefetch_span <= span_bottom_treshold)
                                ? group.async_work_group_copy(buffer_ptr + it.get_local_range(1),
                                                              intermediate_ptr + (dst_w * prefetch_span),
                                                              row_width)
                                : group.async_work_group_copy(buffer_ptr + it.get_local_range(1),
                                                              intermediate_ptr + (dst_w * span_bottom_treshold),
                                                              row_width);
                }
                // else if(prefetch_span == 0){ // could also be else...
                else
                {
                    // prefetch self -- Always safe
                    above_0 = group.async_work_group_copy(buffer_ptr, intermediate_ptr, row_width);
                }

                // SAME AS ABOVE JUST OTHER EVENT
                // Iteration using event set 1
                span--;
                if(span == 0) // Level of which we process
                {
                    // Compute self
                    above_1.wait();
                    out += buffer[(it.get_local_range(1) << 1) + it.get_local_id(1)] * filter[0];
                    break; // Break out to write the result
                }
                prefetch_span--;

                // above_1.wait();

                // COMPUTE HERE
#if COMPUTE_ONE_BY_ONE
                above_1.wait();
                // out += buffer[it.get_local_range(1) * (z << 1) + it.get_local_id(1)] * filter[span];
                out += buffer[(it.get_local_range(1) << 1) + it.get_local_id(1)] * filter[span];

                below_1.wait();
                // out += buffer[it.get_local_range(1) * ((z << 1) + 1) + it.get_local_id(1)] * filter[span];
                out += buffer[(it.get_local_range(1) * 3) + it.get_local_id(1)] * filter[span];
#else

                above_1.wait();
                val_above = buffer[(it.get_local_range(1) << 1) + it.get_local_id(1)];

                below_1.wait();
                val_below = buffer[(it.get_local_range(1) * 3) + it.get_local_id(1)];
                out += (val_above + val_below) * filter[span];

#endif

                if(prefetch_span > 0)
                {
                    // prefetch normal
                    above_1 = (prefetch_span <= write_y)
                                ? group.async_work_group_copy(buffer_ptr + (it.get_local_range(1) << 1),
                                                              intermediate_ptr - (dst_w * prefetch_span),
                                                              row_width)
                                : group.async_work_group_copy(buffer_ptr + (it.get_local_range(1) << 1),
                                                              intermediate_ptr - (dst_w * write_y),
                                                              row_width);

                    below_1 = (prefetch_span <= span_bottom_treshold)
                                ? group.async_work_group_copy(buffer_ptr + it.get_local_range(1) * 3,
                                                              intermediate_ptr + (dst_w * prefetch_span),
                                                              row_width)
                                : group.async_work_group_copy(buffer_ptr + it.get_local_range(1) * 3,
                                                              intermediate_ptr + (dst_w * span_bottom_treshold),
                                                              row_width);
                }
                // else if(prefetch_span == 0){ // could also be else...
                else
                {
                    // prefetch self -- Always safe
                    above_1 = group.async_work_group_copy(
                      buffer_ptr + (it.get_local_range(1) << 1), intermediate_ptr, row_width);
                }
            }

            if(live)
            {
                data_array[0][write_y * dst_w + write_x] = out; // Store synchronously add asycn option for test later
            }

            intermediate_ptr -= dst_w; // Move up center position that we fetch around
        }
#endif // DO_VERT
    }
};

// For debugging remove!
template<bool REMAINDER_COL, bool REMAINDER_ROW>
class BuildOctaveSlidingWindow
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
    BuildOctaveSlidingWindow(syclexp::sampled_image_handle src,
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
#define USE_FULL_SYNC 0
#if USE_FULL_SYNC
        // Ensures all work groups have completed horiz before moving on to vert
        full_sync(sg_region.wg_sync_state, it);

        // if(it.get_local_linear_id() == 0)
        // {
        //     syclexp::printf("WG_ID %d --> Num_wg = %d\n",
        //                     static_cast<int>(it.get_group_linear_id()),
        //                     static_cast<int>(it.get_group_range(0) * it.get_group_range(1)));
        // }
        //
        // horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);
#else
        // Only necessary negbours (top and bottom) are waited on
        horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);
#endif

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
        // if(it.get_global_linear_id() == 0)
        // {
        //     syclexp::printf("write_y = %d intermediate_value = %f \n ", write_y, intermediate[write_y * dst_w]);
        // }
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

        // bottom of our region (safe as write_y has already been bounds checked for below)
        int start_pos = write_y * dst_w + it.get_group(1) * it.get_local_range(1);
        // Bottom row position
        auto intermediate_ptr =
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
            intermediate + start_pos);

        auto raw_ptr = intermediate_ptr.get();
        size_t distance_elems =
          (reinterpret_cast<std::uintptr_t>(raw_ptr) - reinterpret_cast<std::uintptr_t>(intermediate)) / sizeof(float);

        auto buffer_ptr = buffer.template get_multi_ptr<sycl::access::decorated::yes>();

// Changes with loop matching level -- initial is zero always
#if MINIMAL_WINDOW
        int span = d_gauss->inc.span[0] - 1; // As filter[span] is 0 hence no point computing that
#else
        int span = d_gauss->inc.span[0];
#endif

        sycl::group group = it.get_group();

        // NOTE:  Window is always as wide as wide as it.get_local_range(1) but we only load in row_widht wide data
        // This still alows for no bank conflicts and the work-items outside of range can safely do the same as rest
        // on the dono't care data in their location and we only need so safe guard the writes to global memory
        // minimizing the number of if checks we need (and doing more in locstep does not really add performance
        // overhead)

        std::optional<sycl::device_event> above_events[2]; // Avoids using deleted default constructor of device_event
        std::optional<sycl::device_event> below_events[2]; // Avoids using deleted default constructor of device_event

        // Above is always safe in this case as we are at the bottom of our region hence one above always exits

        // Load top of widow
        // Should be span - 1  for multiplication with dst_w as we don't need span row as filter is always zero there
        above_events[0] = group.async_work_group_copy(buffer_ptr, intermediate_ptr - (dst_w * span), row_width);

        // If true it should only do async_work_group copy and nothing else
        bool live = it.get_local_id(1) < row_width;
        // Changes with loop matching level -- initial is zero always
        float* filter = &d_gauss->inc.filter[0];

        int i_max = dst_h - 1 - write_y; // How many pixels we can go down before we are beyond image bounds
        if(span <= i_max)
        {
            // Load in bottom row
            below_events[0] = group.async_work_group_copy(
              buffer_ptr + ((span << 1) * it.get_local_range(1)), intermediate_ptr + (dst_w * span), row_width);
        }
        else
        {
            // CLAMPING (Load from final row of image)
            below_events[0] = group.async_work_group_copy(
              buffer_ptr + ((span << 1) * it.get_local_range(1)), intermediate_ptr + (dst_w * i_max), row_width);
        }

        float out = 0.0f; // Need to add the self part last to get same result as popsift makes big difference on filnal
                          // values due to float inaccuracies
        float g;
        int next_i = span;

        float val_above;
        float val_below;

        std::optional<sycl::device_event> event_center;
        // for(int i = 1; i <= span; ++i)
        // Should be span - 1 as filter[span] is always zero
        for(int i = span; i > 0; --i) // Need to start at largest (smallest filter) for precision
        {
            next_i--;
            if(next_i > 0)
            {
                // Above is known to be safe here aslong as we keep mimimum height of block to largest_span + 1
                // As then we know it's withing bounds for top row and we can omit the check here (we do need it
                // later on when we start sliding the window up)
                // Also safe when y_remainder is less than that as that is at bottom of image and always in image bounds
                above_events[1] = group.async_work_group_copy(
                  buffer_ptr + ((span - next_i) * it.get_local_range(1)), intermediate_ptr - dst_w * next_i, row_width);

                if(i <= i_max)
                {
                    below_events[1] =
                      group.async_work_group_copy(buffer_ptr + ((span + next_i) * it.get_local_range(1)),
                                                  intermediate_ptr + dst_w * next_i,
                                                  row_width);
                }

                else
                {
                    // CLAMPING -- Load from bottom row
                    below_events[1] =
                      group.async_work_group_copy(buffer_ptr + ((span + next_i) * it.get_local_range(1)),
                                                  intermediate_ptr + dst_w * i_max,
                                                  row_width);
                }
            }
            else
            {
                // load in self (center)

                event_center =
                  group.async_work_group_copy(buffer_ptr + (span * it.get_local_range(1)), intermediate_ptr, row_width);
            }
            g = filter[i];

            above_events[0]->wait();
            val_above = buffer[(span - i) * it.get_local_range(1) + it.get_local_id(1)];
            // Could compute and add to out here but I think doing it all in one seems to be more
            // sensible right? But if we are properly memory bound doing it here makes more sense as the
            // extra multliplication don't hurt in that case and more is done earlier

#if COMPUTE_ONE_BY_ONE
            out += (val_above * g);
#endif
            below_events[1]->wait();
            val_below = buffer[(span + i) * it.get_local_range(1) + it.get_local_id(1)];
            // Always in correct position due to copying in case of clamping

#if COMPUTE_ONE_BY_ONE
            out += (val_below * g);
#else
            out += ((val_above + val_below) * g);
#endif
            // ######################################################################################################
            // Second iteration of the same just using different variables for events could do the same with array
            // of events and do mod to figoure out which one to use but this should be less expensive than doing mod
            // (though less readable)
            // ######################################################################################################

            // Next iteration
            if(next_i <= 0) // Exit check
                break;

            i = next_i; // Not sure if is faster ?
            next_i--;
            // Next iteration with swaped event variables

            // if(next_i <= span)
            if(next_i > 0)
            {
                // Always safe
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
                    // As we are moving inwards we dono't already have loaded the values so need to load from
                    // intermeidate directly
                    above_events[0] =
                      group.async_work_group_copy(buffer_ptr + ((span + next_i) * it.get_local_range(1)),
                                                  intermediate_ptr + dst_w * i_max,
                                                  row_width);
                }
            }
            else
            {
                // load in self (center)

                event_center =
                  group.async_work_group_copy(buffer_ptr + (span * it.get_local_range(1)), intermediate_ptr, row_width);
            }

            g = filter[i];

            above_events[1]->wait();
            val_above = buffer[(span - i) * it.get_local_range(1) + it.get_local_id(1)];

            // Don't need to ensure clamping as we have copied clamped values to their correct position in window to
            // avoid having to do if checks in loop below

#if COMPUTE_ONE_BY_ONE
            out += (val_above * g);
#endif
            below_events[1]->wait();

            val_below = buffer[(span + i) * it.get_local_range(1) + it.get_local_id(1)];
#if COMPUTE_ONE_BY_ONE
            out += (val_below * g);
#else
            out += ((val_above + val_below) * g);
#endif

            // Now we have done second iteration and next to wait is above and below 1 and prefetch 2 so we iterate
        }
        // Do we want to do write async aswell? Or will that just result in worse performance? mby test

        event_center->wait();

        float val = buffer[span * it.get_local_range(1) + it.get_local_id(1)];
        out += (val * filter[0]);
        if(live)
        {
            data_array[0][write_y * dst_w + write_x] = out; // Store synchronously add asycn option for test later
        }

        std::optional<sycl::device_event> next_row_event; // Avoids using deleted default constructor of device_event
        int next_row_fetch = write_y - (span + 1);        // Set it to top row aswell so we can just decrement the value

        if(next_row_fetch >= 0)
        {
            // if greater than zero we can load next row
            intermediate_ptr -= (span + 1) * dst_w;
            next_row_event = group.async_work_group_copy(
              buffer_ptr + ((span << 1) + 1) * it.get_local_range(1), intermediate_ptr, row_width);
        }
        else
        {
            // we need to copy from local memory to the location to avoid if's later
            buffer[((span << 1) + 1) * it.get_local_range(1) + it.get_local_id(1)] = buffer[it.get_local_id(1)];
        }

// now we have intermediate_ptr at top and we can load that into the free row

// int i_max = dst_h - write_y - 1;
// if (write_y span + 1
#define old_way 0

        write_y--;              // decrement write_y as we have done the first iteration
        int free = (span << 1); // as we have already done it when free is at (span<<1) +  1 location (final pos)
        int end_pos = (it.get_global_id(0) * sg_region.height);
        // for(; write_y >= end_pos; write_y--)
        while(true)
        {
            // for(int free = (span << 1); free >= 0; free--)
            for(; free >= 0; free--)
            // can stop and start at these position as then next loop it will be wrong
            {
                // inner loop that moves the prefetch location

                int row_pos = free - (span + 1);
                int self_loop_size;
                if(row_pos >= 0)
                {
#if old_way
                    val = buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)];
                    out = val * filter[0]; // reset by setting
#endif
                    // Do last

                    // compute size of loop around self and implicitly loop size of around free
                    self_loop_size = row_pos; // dist to top
                }
                else
                {
                    // wrap around
                    row_pos += (span << 1) + 2;

#if old_way
                    // Do last
                    val = buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)];
                    out = val * filter[0]; // reset by setting
#endif

                    // compute size of loop around self and implicitly loop size of around free
                    self_loop_size = ((span << 1) + 1) - row_pos; // dist to botom
                }
                // todo: folow outline below and check that the whole logic works with wraparound and such

                // now we have done self (the row we are writing to where the weight is only used once)
                // now we need to do the span on each side

                //

                // do the loop around row_pos

#if !old_way
                out = 0.0f; // Need to rest as we cant reset by setting
                // #########################################
                // ############ NEW LOOP START #############
                // #########################################

                if(it.get_global_linear_id() == 0)
                {
#if DEBUG
                    syclexp::printf("Self_loop_size = %d  -- row_pos = %d -- free = %d -- write_y = %d\n",
                                    self_loop_size,
                                    row_pos,
                                    free,
                                    write_y);
#endif
                }

                // Need to be done early as it's the first accessed value -- Slow for first iteration of while loop rest
                // should be fine
                if(next_row_event.has_value())
                {
                    next_row_event->wait();
                }

                // load next_row_value here
                write_y--; // decrement here (we need to add one when writing result) this decrement is a bit early
                           // as it's for next already while we have not written current yet

                if(write_y >= end_pos)
                {
                    // load in next row either async or copy from local to work as clamp
                    if((write_y - (span + 1)) >= 0)
                    {
                        intermediate_ptr -= dst_w; // move to row above

                        next_row_event = group.async_work_group_copy(
                          buffer_ptr + free * it.get_local_range(1), intermediate_ptr, row_width);
                    }
                    else
                    {
                        // clamp to edge by copying data from prev row(row below)
                        // if multiple needed it works as a chain and as intended

                        // NOTE: Could make one iteration outside of the loop so that we don't need this if...
                        int row_below = free + 1;
                        if(row_below <= (span << 1) + 1) // final row of widow
                        {
                            buffer[free * it.get_local_range(1) + it.get_local_id(1)] =
                              buffer[row_below * it.get_local_range(1) + it.get_local_id(1)];

                            // Might be good to unset the event so that we don't wait on Old event (not sure what's
                            // better)
                            next_row_event.reset();
                        }
                        else
                        {
                            // free is row at bottom of window so we need to load from top as that is whats below it
                            // when we wrap around
                            buffer[free * it.get_local_range(1) + it.get_local_id(1)] = buffer[it.get_local_id(1)];

                            next_row_event.reset();
                        }
                    }
                }

                int offset = span;

                // for(int i = span - self_loop_size; i > 0; --i)
#if DEBUG

                if(it.get_global_linear_id() == 0)
                {
                    syclexp::printf("\tFree loop, i = %d \n", span - self_loop_size);
                }
#endif
                // for(int i = span - self_loop_size; i > 0; --i)
                for(int i = 1; i <= (span - self_loop_size); ++i) // Closes to free is farthest from self
                {
                    // loop around free row
                    val_above = buffer[(free - i) * it.get_local_range(1) + it.get_local_id(1)];

#if COMPUTE_ONE_BY_ONE
                    out += (val_above * filter[offset]);
#endif
                    val_below = buffer[(free + i) * it.get_local_range(1) + it.get_local_id(1)];

#if COMPUTE_ONE_BY_ONE
                    out += (val_below * filter[offset]);
#else

                    out += ((val_above + val_below) * filter[offset]);
                    // if(it.get_global_linear_id() == 0)
                    // {
                    //     syclexp::printf("out = %f <-- (val_above=%f + val_below=%f) * filter[%d] = %f NORMAL\n",
                    //                     out,
                    //                     val_above,
                    //                     val_below,
                    //                     offset,
                    //                     filter[offset]);
                    // }

#endif

#if DEBUG
                    if(it.get_global_linear_id() == 0)
                    {
                        syclexp::printf("\t\ti = %d -> v_above = %f -- v_below = %f -> added_to_out = %f -> "
                                        "Resulting_out = %f - filter[%d] = %.8f\n",
                                        i,
                                        val_above,
                                        val_below,
                                        ((val_above + val_below) * filter[offset]),
                                        out,
                                        offset,
                                        filter[offset]);
                    }
#endif
                    offset--;
                }
                // Offset will be equal to self_loop_size here

#if DEBUG
                if(it.get_global_linear_id() == 0)
                {
                    syclexp::printf("\tself_loop offset_start=%d \n", offset);
                }
#endif

                for(; offset > 0; --offset)
                {
                    // load value around self
                    val_above = buffer[(row_pos - offset) * it.get_local_range(1) + it.get_local_id(1)];
#if COMPUTE_ONE_BY_ONE
                    out += (val_above * filter[offset]);
#endif
                    val_below = buffer[(row_pos + offset) * it.get_local_range(1) + it.get_local_id(1)];

#if COMPUTE_ONE_BY_ONE
                    out += (val_below * filter[offset]);
#else
                    out += ((val_above + val_below) * filter[offset]);
                    // if(it.get_global_linear_id() == 0)
                    // {
                    //     syclexp::printf("out = %f <-- (val_above=%f + val_below=%f) * filter[%d] = %f NORMAL\n",
                    //                     out,
                    //                     val_above,
                    //                     val_below,
                    //                     offset,
                    //                     filter[offset]);
                    // }

#endif

#if DEBUG
                    if(it.get_global_linear_id() == 0)
                    {
                        syclexp::printf("\t\to = %d -> v_above = %f -- v_below = %f -> added_to_out = %f -> "
                                        "Resulting_out = %f - filter[%d] = %.8f\n",
                                        offset,
                                        val_above,
                                        val_below,
                                        ((val_above + val_below) * filter[offset]),
                                        out,
                                        offset,
                                        filter[offset]);
                    }
#endif
                    // using one out is same as two atleast from sample test (with respect to precision)
                }

                // Need to do self

                // val = buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)];
                // out = val * filter[0]; // reset by setting

                // Compute self last to get same value as popsift
                out += (buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)] * filter[0]);
                // if(it.get_global_linear_id() == 0)
                // {
                //     syclexp::printf("out = %f -- val = %f -- filter[%d] = %f NORMAL\n",
                //                     out,
                //                     buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)],
                //                     0,
                //                     filter[0]);
                // }

                // #########################################
                // ############ NEW LOOP END ###############
                // #########################################

#endif

#if old_way
                // #########################################
                // ############ OLD LOOPY LOOP START #######
                // #########################################

#if DEBUG
                if(it.get_global_linear_id() == 0)
                {
                    syclexp::printf("Self_loop_size = %d  -- row_pos = %d -- free = %d -- write_y = %d\n",
                                    self_loop_size,
                                    row_pos,
                                    free,
                                    write_y);
                }
#endif

                // by copying for clamping we don't need any boundary checks for these two loops
                int offset = 1;

#if DEBUG
                if(it.get_global_linear_id() == 0)
                {
                    syclexp::printf("\tself_loop offset_start=%d \n", offset);
                }
#endif
                for(; offset < self_loop_size; ++offset)
                {
                    // load value around self
                    val_above = buffer[(row_pos - offset) * it.get_local_range(1) + it.get_local_id(1)];
                    val_below = buffer[(row_pos + offset) * it.get_local_range(1) + it.get_local_id(1)];

                    // out += (val_above * filter[offset]);
                    // out += (val_below * filter[offset]);
                    out += ((val_above + val_below) * filter[offset]);
                    // using one out is same as two atleast from sample test (with respect to precision)

#if DEBUG
                    if(it.get_global_linear_id() == 0)
                    {
                        syclexp::printf("\t\to = %d -> v_above = %f -- v_below = %f -> added_to_out = %f -> "
                                        "Resulting_out = %f - filter[%d] = %.8f\n",
                                        offset,
                                        val_above,
                                        val_below,
                                        ((val_above + val_below) * filter[offset]),
                                        out,
                                        offset,
                                        filter[offset]);
                    }
#endif
                }
                // moving final iteration out of loop so that we can wait on prev row load first
                // final row could either be final iteration for self loop or final iteration for free loop so
                // waiting here ensures it's loaded and avoids if inside of loop. should also in most cases be late
                // enough that it won't hamper performance too much (could do if instead inside of loop but i think
                // that's worse) // in some cases it will be bad when self is at top of window all is around free
                // hance it's done straight away
                if(next_row_event.has_value())
                {
                    next_row_event->wait();
                }

                // load next_row_value here
                write_y--; // decrement here (we need to add one when writing result) this decrement is a bit early
                           // as it's for next already while we have not written current yet

                if(write_y >= end_pos)
                {
                    // load in next row either async or copy from local to work as clamp
                    if((write_y - (span + 1)) >= 0)
                    {
                        intermediate_ptr -= dst_w; // move to row above

                        next_row_event = group.async_work_group_copy(
                          buffer_ptr + free * it.get_local_range(1), intermediate_ptr, row_width);
                    }
                    else
                    {
                        // clamp to edge by copying data from prev row(row below)
                        // if multiple needed it works as a chain and as intended
                        int row_below = free + 1;
                        if(row_below <= (span << 1) + 1) // final row of widow
                        {
                            buffer[free * it.get_local_range(1) + it.get_local_id(1)] =
                              buffer[row_below * it.get_local_range(1) + it.get_local_id(1)];
                        }
                        else
                        {
                            // free is row at bottom of window so we need to load from top as that is whats below it
                            // when we wrap around
                            buffer[free * it.get_local_range(1) + it.get_local_id(1)] = buffer[it.get_local_id(1)];
                        }
                    }
                }

                // offset should be equal to self_loop_size now as that is the value when it terminated loop above
                if(self_loop_size != 0)
                {
                    val_above = buffer[(row_pos - self_loop_size) * it.get_local_range(1) + it.get_local_id(1)];
                    val_below = buffer[(row_pos + self_loop_size) * it.get_local_range(1) + it.get_local_id(1)];

                    // out += (val_above * filter[offset]);
                    // out += (val_below * filter[offset]);
                    out += ((val_above + val_below) * filter[offset]);

                    // increment offset here so that we are at correct offset when this runs and when we only loop
                    // around free

#if DEBUG
                    if(it.get_global_linear_id() == 0)
                    {
                        syclexp::printf("\t\to = %d -> v_above = %f -- v_below = %f -> added_to_out = %f -> "
                                        "Resulting_out = %f - filter[%d] = %.8f\n",
                                        offset,
                                        val_above,
                                        val_below,
                                        ((val_above + val_below) * filter[offset]),
                                        out,
                                        offset,
                                        filter[offset]);
                    }
#endif

                    offset++;
                }

#if DEBUG
                if(it.get_global_linear_id() == 0)
                {
                    syclexp::printf("\tFree loop, i = %d \n", span - self_loop_size);
                }
#endif
                for(int i = span - self_loop_size; i > 0; --i)
                {
                    // loop around free row
                    val_above = buffer[(free - i) * it.get_local_range(1) + it.get_local_id(1)];
                    val_below = buffer[(free + i) * it.get_local_range(1) + it.get_local_id(1)];
                    // out += (val_above * filter[offset]);
                    // out += (val_below * filter[offset]);
                    out += ((val_above + val_below) * filter[offset]);

#if DEBUG
                    if(it.get_global_linear_id() == 0)
                    {
                        syclexp::printf("\t\ti = %d -> v_above = %f -- v_below = %f -> added_to_out = %f -> "
                                        "Resulting_out = %f - filter[%d] = %.8f\n",
                                        i,
                                        val_above,
                                        val_below,
                                        ((val_above + val_below) * filter[offset]),
                                        out,
                                        offset,
                                        filter[offset]);
                    }
#endif

                    offset++; // only increment at end so that when ours is at top and bottom of window we use
                              // correct filter
                }

                // #########################################
                // ############ OLD LOOPY LOOP END #########
                // #########################################

#endif

                if(live)
                {
                    // we need to pluss one to write_y as we did decrement it a bit early
                    // so that we are writing to correct position

#if DEBUG
                    if(it.get_global_linear_id() == 0)
                    {
                        syclexp::printf("\t\t\twrite_y = %d --> out = %f\n", write_y, out);
                    }
#endif

                    data_array[0][(write_y + 1) * dst_w + write_x] = out; // synchronous setting

                    // if(it.get_global_linear_id() == 0)
                    // {
                    //     syclexp::printf("out = %f -- FINAL -- write_y = %d\n\n", out, write_y + 1);
                    // }

                    // if(it.get_global_linear_id() == 0)
                    // {
                    //     syclexp::printf("out = %f   - p2\n", out);
                    // }
                }

                if(write_y < end_pos) // next check is here we currently at write_y + 1
                {
                    break; // we are done // outer loop will also exit at this point
                }
            }
            if(write_y < end_pos)
            {
                break; // break out of otuer loop
            }
            // otherwise reset free and continue

            // set free to final pos as we have not had one iteration outside of the loop now
            free = (span << 1) + 1;
        }

#if false // If we are continuing on we need synchronization
#if DO_HORIZ
        vert_sync_for_horiz(sg_region.wg_sync_state, it, 2);
#else
        // Set to one as we have not done horiz and hence can't get to that
        vert_sync_for_horiz(sg_region.wg_sync_state, it, 1);
#endif
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

#define normalOctave true

        if(true)
        {
            printf("DOING BUILD OCTAVE SIMPLE\n\n");
            // return _device_queue.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
            //                                   {d_gauss_write, img_write, sg_region._zeroed_event},
            //                                   normalizedSource::BuildOcaveSimple(base->getInputImage(),
            //                                                                      oct_obj.getDataArray(),
            //                                                                      oct_obj.getDogArray(),
            //                                                                      oct_obj.getIntermediate(),
            //                                                                      _d_gauss,
            //                                                                      sg_region.sg_block,
            //                                                                      width,
            //                                                                      height,
            //                                                                      shift,
            //                                                                      _levels));

            // TODO: MAKE IT USE ONE VARIABLE THAT WE NNED TO RESET TO ZERO. THAT IS THE REASON WHY IT IS NOT WORKING
            // NOT MOST LIKELY SO SEEMS TO WORK WEHN WAVE IS 1 AS EXPECTED... NEED TO REDUCE REGISTER USAGE SO THAT WE
            // HAVE ENOUGHT TO DO THE SMALLER SYNCRONIZATION BUT MYB GLOBAL IS GOOD ENOUGH
            // --> Test using weaker than the current strongest atomic ref for the full_sync and try to find way to
            // reduce register usage

            // BUG: DOES NOT WORK ON SECOND RUN(IMAGE) DUE TO NOT RESETING THE sg_region memory used to do atomic
            // counting hence counter is reached instantly on second go and does not do shit

            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // auto buffer = sycl::local_accessor<float, 1>(sg_region.local_mem_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
                                 normalizedSource::BuildOctaveSimple(base->getInputImage(),
                                                                     oct_obj.getDataArray(),
                                                                     oct_obj.getDogArray(),
                                                                     oct_obj.getIntermediate(),
                                                                     _d_gauss,
                                                                     sg_region.sg_block,
                                                                     width,
                                                                     height,
                                                                     shift,
                                                                     _levels));
            });
        }

        else if(col && row)
        {
            printf("We doing col and row whop whop\n");
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

#if normalOctave
                                 normalizedSource::BuildOctave<true, true>(base->getInputImage(),
#else
                                 normalizedSource::BuildOctaveSlidingWindow<true, true>(base->getInputImage(),
#endif
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
#if normalOctave
                                 normalizedSource::BuildOctave<true, true>(base->getInputImage(),
#else
                                 normalizedSource::BuildOctaveSlidingWindow<true, true>(base->getInputImage(),
#endif
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
#if normalOctave
                                 normalizedSource::BuildOctave<true, true>(base->getInputImage(),
#else
                                 normalizedSource::BuildOctaveSlidingWindow<true, true>(base->getInputImage(),
#endif
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
                // BuildOctaveSlidingWindow
#if normalOctave
                                 normalizedSource::BuildOctave<true, true>(base->getInputImage(),
#else
                                 normalizedSource::BuildOctaveSlidingWindow<true, true>(base->getInputImage(),
#endif
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

#if false
// OLD loop code for window for reference... DELETE WHEN YOU CAN

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
                    // Can't do that because of precision need to be done last...
                    // val = buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)];
                    // out = val * filter[0]; // Reset by setting

                    // compute size of loop around self and implicitly loop size of around free
                    self_loop_size = row_pos; // Dist to top
                }
                else
                {
                    // wrap around
                    row_pos += (span << 1) + 2;

                    // Can't do that because of precision need to be done last...
                    // val = buffer[row_pos * it.get_local_range(1) + it.get_local_id(1)];
                    // out = val * filter[0]; // Reset by setting

                    // compute size of loop around self and implicitly loop size of around free
                    self_loop_size = ((span << 1) + 1) - row_pos; // Dist to botom
                }

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

                    offset++; // only increment at end so that when ours is at top and bottom of window we use
                              // correct filter
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
