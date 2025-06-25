
int start_pos = write_y * dst_w + it.get_group(1) * it.get_local_range(1);

// Bottom row position
auto intermediate_ptr =
  sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(intermediate +
                                                                                                    start_pos);

auto buffer_ptr = buffer.template get_multi_ptr<sycl::access::decorated::yes>();

int span = d_gauss->inc.span[0];

sycl::group group = it.get_group();

// Load into middle of window
sycl::device_event evt_center =
  group.async_work_group_copy(intermediate_ptr, buffer_ptr + (span * row_width), row_width);

// Changes with loop matching level -- initial is zero always
float* filter = &d_gauss->inc.filter[0];

// Events for loading in next and waiting on prev to do compute
// sycl::device_event above_1_evt;
sycl::device_event above_2_evt = sycl::device_event();
sycl::device_event below_1_evt = sycl::device_event();
sycl::device_event below_2_evt = sycl::device_event();

sycl::device_event above_1_evt =
  group.async_work_group_copy(intermediate_ptr - dst_w, buffer_ptr + ((span - 1) * row_width), row_width);

if(write_y + 1 < dst_h)
{
    below_1_evt =
      group.async_work_group_copy(intermediate_ptr + dst_w, buffer_ptr + ((span + 1) * row_width), row_width);
}

// Need clamping logic

evt_center.wait();

float val = buffer[span * row_width + it.get_local_id(1)];
float out = val * filter[0];
float g;

int i_max = dst_h - write_y - 1;
for(int i = 1; i <= span; ++i)
{
    int next_i = i + 1;
    if(next_i <= span)
    {
        // Above is known to be safe here aslong as we keep mimimum height of block to largest_span + 1
        // As then we know it's withing bounds for top row and we can omit the check here (we do need it later
        // on when we start sliding the window up)
        above_2_evt = group.async_work_group_copy(
          intermediate_ptr - dst_w * next_i, buffer_ptr + ((span - next_i) * row_width), row_width);

        if(i <= i_max)
        {
            below_2_evt = group.async_work_group_copy(
              intermediate_ptr + dst_w * next_i, buffer_ptr + ((span + next_i) * row_width), row_width);
        }
    }
    g = filter[i];
    above_1_evt.wait();
    int val_above = buffer[(span - i) * row_width + it.get_local_id(1)];

    below_1_evt.wait();
    // Clamp to edge
    int val_below = i <= i_max ? buffer[(span + i) * row_width + it.get_local_id(1)]
                               : buffer[(span + i_max) * row_width + it.get_local_id(1)];

    out += ((val_above + val_below) * g);

    // Second iteration of the same just using different variables for events could do the same with array of
    // events and do mod to figoure out which one to use but this should be less expensive than doing mod
    // (though less readable)

    // Next iteration
    if(next_i > span) // Exit check
        break;

    i = next_i; // Not sure if incrementing is faster ?
    next_i++;
    // Next iteration with swaped event variables

    if(next_i <= span)
    {
        above_1_evt = group.async_work_group_copy(
          intermediate_ptr - dst_w * next_i, buffer_ptr + ((span - next_i) * row_width), row_width);

        if(i <= i_max)
        {
            below_1_evt = group.async_work_group_copy(
              intermediate_ptr + dst_w * next_i, buffer_ptr + ((span + next_i) * row_width), row_width);
        }
    }

    g = filter[i];

    above_2_evt.wait();
    val_above = buffer[(span - i) * row_width + it.get_local_id(1)];

    below_2_evt.wait();
    // Clamp to edge
    val_below = i <= i_max ? buffer[(span + i) * row_width + it.get_local_id(1)]
                           : buffer[(span + i_max) * row_width + it.get_local_id(1)];

    out += ((val_above + val_below) * g);
    // Now we have done second iteration and next to wait is above and below 1 and prefetch 2 so we iterate
}
