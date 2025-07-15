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
        const int write_x = it.get_global_id(1);
        int write_y = it.get_global_id(0) * sg_region.height; // Changes in normal block
#if DO_VERT

#if !DO_HORIZ
        // So it is the same it would have been if we were doing horiz
        write_y = sycl::min(write_y + sg_region.height - 1, dst_h - 1);
#endif

        const int row_width = [&]() {
            if constexpr(REMAINDER_ROW)
            {
                if(it.get_group(1) == (it.get_group_range(1) - 1))
                {
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

        const bool live = [&]() {
            if constexpr(REMAINDER_ROW)
            {
                return it.get_local_id(1) < row_width;
            }
            else
            {
                // There is no remainder hence all are full rows
                return true;
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

        // auto inter =
        //   sycl::make_ptr<sycl::access::address_space::global_space, sycl::access::decorated::yes>(intermediate);
        // auto intermeidate_ptr = inter.template get_multi_ptr<sycl::access::decorated::yes>();
        auto buffer_ptr = buffer.template get_multi_ptr<sycl::access::decorated::yes>();

        float* filter = &d_gauss->inc.filter[0]; // For level 0
        sycl::group group = it.get_group();

        // NOTE:  Window is always as wide as wide as it.get_local_range(1) but we only load in row_widht wide data
        // This still alows for no bank conflicts and the work-items outside of range can safely do the same as rest
        // on the dono't care data in their location and we only need so safe guard the writes to global memory
        // minimizing the number of if checks we need (and doing more in locstep does not really add performance
        // overhead)

        // Two async rows and one current row for both above and below (6 total)
        std::optional<sycl::device_event> above_events[3]; // Avoids using deleted default constructor of device_event
        std::optional<sycl::device_event> below_events[3]; // Avoids using deleted default constructor of device_event

        std::optional<sycl::device_event> self; // Self center load

        // Above is always safe in this case as we are at the bottom of our region hence one above always exits

        int end_pos = (it.get_global_id(0) * sg_region.height);
        float out;

        // If not live it does everything besides write as it's comoputing non-sense
#if MINIMAL_WINDOW
        int span_width = d_gauss->inc.span[0] - 1;
#else
        int span_width = d_gauss->inc.span[0];
#endif

        for(; write_y >= end_pos; --write_y)
        {
            int span = span_width;

            // Need to modify intermediate pointer
            // How many pixels we can go down before we are beyond image bounds
            int span_bottom_treshold = dst_h - 1 - write_y;

            // I'ts just wirte_y
            // int span_top_treshold = write_y; // How many pixels we can go down before we are beyond image bounds

            // Start prefetching around write_y the outer most rows then second then third before entering the loop and
            // waiting on them

            // NOTE: Assumes that span cannot be less than 4
            // -> if that is not the case it will do unnecessary loads but won't cause it to compute wrong value or fail

#if true
            // if(write_y - span >= 0)
            // Store initial in second for better mathing with loop
            if(span <= write_y) // if greater we are out of bounds (above image)
            {
                // load outer most row pair
                above_events[2] = group.async_work_group_copy(
                  buffer_ptr + 4 * it.get_local_range(1), intermediate_ptr - (dst_w * span), row_width);
            }
            else
            {
                // load top row (clamp to edge)
                above_events[2] = group.async_work_group_copy(
                  buffer_ptr + 4 * it.get_local_range(1), intermediate_ptr - (dst_w * write_y), row_width);
            }

            // Second row of local memory (use full row even when we don't load full row for bank conflict avoidance and
            // avoid branching)
            if(span <= span_bottom_treshold)
            {
                below_events[2] = group.async_work_group_copy(
                  buffer_ptr + 5 * it.get_local_range(1), intermediate_ptr + (dst_w * span), row_width);
            }
            else
            {
                below_events[2] = group.async_work_group_copy(
                  buffer_ptr + 5 * it.get_local_range(1), intermediate_ptr + (dst_w * span_bottom_treshold), row_width);
            }

            // second outermost row pair
            if((span - 1) <= write_y)
            {
                above_events[0] =
                  group.async_work_group_copy(buffer_ptr, intermediate_ptr - (dst_w * (span - 1)), row_width);
            }
            else
            {
                above_events[0] =
                  group.async_work_group_copy(buffer_ptr, intermediate_ptr - (dst_w * write_y), row_width);
            }

            // if(write_y + (span - 1) < dst_h)
            if((span - 1) <= span_bottom_treshold)
            {
                below_events[0] = group.async_work_group_copy(
                  buffer_ptr + it.get_local_range(1), intermediate_ptr + (dst_w * (span - 1)), row_width);
            }
            else
            {
                below_events[0] = group.async_work_group_copy(
                  buffer_ptr + it.get_local_range(1), intermediate_ptr + (dst_w * span_bottom_treshold), row_width);
            }

            // Third outmost row pair
            if((span - 2) <= write_y)
            {
                above_events[1] = group.async_work_group_copy(
                  buffer_ptr + 2 * it.get_local_range(1), intermediate_ptr - (dst_w * (span - 2)), row_width);
            }
            else
            {
                above_events[1] = group.async_work_group_copy(
                  buffer_ptr + 2 * it.get_local_range(1), intermediate_ptr - (dst_w * write_y), row_width);
            }

            if((span - 2) <= span_bottom_treshold)
            {
                below_events[1] = group.async_work_group_copy(
                  buffer_ptr + 3 * it.get_local_range(1), intermediate_ptr + (dst_w * (span - 2)), row_width);
            }
            else
            {
                below_events[1] = group.async_work_group_copy(
                  buffer_ptr + 3 * it.get_local_range(1), intermediate_ptr + (dst_w * span_bottom_treshold), row_width);
            }

#if COMPUTE_ONE_BY_ONE
            above_events[2]->wait();
            out = buffer[(4 * it.get_local_range(1)) + it.get_local_id(1)] * filter[span]; // Reset by setting

            below_events[2]->wait();
            out += buffer[(5 * it.get_local_range(1)) + it.get_local_id(1)] * filter[span];
#else
            // group.wait_for(above_events[2], below_events[2]);
            above_events[2]->wait();
            below_events[2]->wait();

            float val_above = buffer[(4 * it.get_local_range(1)) + it.get_local_id(1)];
            float val_below = buffer[(5 * it.get_local_range(1)) + it.get_local_id(1)];
            out = (val_above + val_below) * filter[span]; // reset by setting

#endif
#endif

#if true
            if((span - 3) <= write_y)
            {
                above_events[2] = group.async_work_group_copy(
                  buffer_ptr + 4 * it.get_local_range(1), intermediate_ptr - (dst_w * (span - 3)), row_width);
            }
            else
            {
                above_events[2] = group.async_work_group_copy(
                  buffer_ptr + 4 * it.get_local_range(1), intermediate_ptr - (dst_w * write_y), row_width);
            }

            if((span - 3) <= span_bottom_treshold)
            {
                below_events[2] = group.async_work_group_copy(
                  buffer_ptr + 5 * it.get_local_range(1), intermediate_ptr + (dst_w * (span - 3)), row_width);
            }
            else
            {
                below_events[2] = group.async_work_group_copy(
                  buffer_ptr + 5 * it.get_local_range(1), intermediate_ptr + (dst_w * span_bottom_treshold), row_width);
            }

            while(span > 0) // loop over current to compute the out value // Could replace this one with shared memory
            {
#pragma unroll
                for(int z = 0; z < 3; ++z)
                {
                    // Three iterations for the buffering
                    span--;
                    if(span < 1) // Level of which we process
                    {
                        break; // Break out to write the result
                    }

#if COMPUTE_ONE_BY_ONE
                    above_events[z]->wait();
                    out += buffer[it.get_local_range(1) * (z << 1) + it.get_local_id(1)] * filter[span];

                    below_events[z]->wait();
                    out += buffer[it.get_local_range(1) * ((z << 1) + 1) + it.get_local_id(1)] * filter[span];
#else
                    // group.wait_for(above_events[z], below_events[z]);
                    above_events[z]->wait();
                    below_events[z]->wait();

                    float val_above = buffer[it.get_local_range(1) * (z << 1) + it.get_local_id(1)];
                    float val_below = buffer[it.get_local_range(1) * ((z << 1) + 1) + it.get_local_id(1)];
                    out += (val_above + val_below) * filter[span];
#endif

                    int prefetch_span = span - 3;
                    if(prefetch_span > 0) // values we need
                    {
                        if(prefetch_span <= write_y)
                        {
                            above_events[z] = group.async_work_group_copy(buffer_ptr + z * it.get_local_range(1),
                                                                          intermediate_ptr - (dst_w * prefetch_span),
                                                                          row_width);
                        }
                        else
                        {
                            above_events[z] = group.async_work_group_copy(
                              buffer_ptr + z * it.get_local_range(1), intermediate_ptr - (dst_w * write_y), row_width);
                        }

                        if(prefetch_span <= span_bottom_treshold)
                        {
                            below_events[z] = group.async_work_group_copy(buffer_ptr + (z + 1) * it.get_local_range(1),
                                                                          intermediate_ptr + (dst_w * prefetch_span),
                                                                          row_width);
                        }
                        else
                        {
                            below_events[z] =
                              group.async_work_group_copy(buffer_ptr + (z + 1) * it.get_local_range(1),
                                                          intermediate_ptr + (dst_w * span_bottom_treshold),
                                                          row_width);
                        }
                    }
                    else if(prefetch_span == 0)
                    {
                        // load self
                        above_events[z] = group.async_work_group_copy(
                          buffer_ptr + (z << 1) * it.get_local_range(1), intermediate_ptr, row_width);
                    }
                }
            }
#endif
// Store out value to correct position
#if true
            int self_pos = (span_width - 1) % 3;
            above_events[self_pos]->wait(); // Wait on self

            // Add self
            out += buffer[it.get_local_range(1) * (self_pos << 1) + it.get_local_id(1)] * filter[span];

            if(live)
            {
                data_array[0][write_y * dst_w + write_x] = out; // Store synchronously add asycn option for test later
            }
#endif
            // something wrong about the way we modify the intermeidaet_ptr
            // GO up one row and do compute there
            intermediate_ptr -= dst_w; // Move up center position that we fetch around
        }
#endif // DO_VERT
    }
};
