static inline unsigned int extrema_count(bool indicator, int* extrema_counter, sycl::nd_item<3>& it)
{
    sycl::sub_group sub_group = it.get_sub_group();
    int ct = sycl::reduce_over_group(sub_group, indicator ? 1 : 0, sycl::plus<>());

    int write_index;
    if(sub_group.leader()) // is always work-item with local_id 0 in the sub_group
    {
        write_index = sycl::atomic_ref<int,
                                       sycl::memory_order_relaxed,
                                       sycl::memory_scope_device,
                                       sycl::access::address_space::global_space>(*extrema_counter) += ct;
    }

    // Give value before add to every work-items in sub_group
    write_index = sycl::group_broadcast(sub_group, write_index, 0);

    // Add sum of threads from [zero, self] so including itself and all before that hasd indicator 1
    // Used to increment write index so that every true indicator writes to different indecies
    write_index += sycl::inclusive_scan_over_group(sub_group, indicator ? 1 : 0, sycl::plus<>());

    return write_index;
}

static inline unsigned int extrema_count(bool indicator, int* extrema_counter, sycl::nd_item<3>& it)
{
    sycl::sub_group sub_group = it.get_sub_group();

    int item_offset = sycl::inclusive_scan_over_group(sub_group, indicator ? 1 : 0, sycl::plus<>());
    int write_index = 0; // Must be set to zero (Incorrect result otherwise)
    if(sub_group.get_local_id()[0] == sub_group.get_local_range()[0] - 1) // Only last work-item has complete count
    {
        // The atomic add returns the old value in extrema_coutner before the addition which is considered the base
        // As each thread uses this and adds to it's own counter (write_index) how many of the threads in the
        // sub-group before it had it's indicator to true
        write_index = sycl::atomic_ref<int,
                                       sycl::memory_order_relaxed,
                                       sycl::memory_scope_device,
                                       sycl::access::address_space::global_space>(*extrema_counter) += item_offset;
    }
    // work-item 0 broadcassts to all other same as leader work-item
    // everyone now get's the base value that they can add to
    int base = sycl::group_broadcast(sub_group, write_index, 0);

    return base + item_offset - (indicator ? 1 : 0);
}
