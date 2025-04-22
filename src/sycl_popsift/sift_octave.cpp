#include "sycl_popsift/sift_octave.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/bindless_helpers.hpp"
#include "sycl_popsift/common/debug_macros.hpp"

#include <sstream>

namespace popsift {

namespace syclexp = sycl::ext::oneapi::experimental;

/*************************************************************
 * OctaveBase
 *************************************************************/

OctaveBase::OctaveBase(sycl::queue Q)
  : _device_queue(Q)
{}

void OctaveBase::alloc(const Config& conf, int width, int height, int levels)
{
    _max_w = _w = width;
    _max_h = _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    if(_level_complete_events)
        delete[] _level_complete_events;

    _level_complete_events = new sycl::event[levels];

    alloc_arrays();
}

/*************************************************************
 * Octave -- USM
 *************************************************************/

Octave::Octave(sycl::queue Q)
  : OctaveBase(Q)
{}

// void Octave::alloc(const Config& conf, int width, int height, int levels)
// {
//     _max_w = _w = width;
//     _max_h = _h = height;
//     _levels = levels;
//
//     _w_grid_divider = float(_w) / conf.getFilterGridSize();
//     _h_grid_divider = float(_h) / conf.getFilterGridSize();
//
//     _level_complete_events = new sycl::event[levels];
//
//     alloc_arrays();
// }

void Octave::alloc_arrays()
{
    // Could use shared for these ones aswell probs same end result and cleaner code

    _data_array_host =
      popsift::common::new_hostT<float*>(_levels, __FILE__, __LINE__, "Host allocation for data array failed");
    _dog_array_host =
      popsift::common::new_hostT<float*>(_levels - 1, __FILE__, __LINE__, "Host allocation for DoG array failed");

    _data_array = popsift::sycl_common::malloc_devT<float*>(
      _levels, __FILE__, __LINE__, "Device allocation for data array failed", _device_queue);

    _dog_array = popsift::sycl_common::malloc_devT<float*>(
      _levels - 1, __FILE__, __LINE__, "Device allocation for DoG array failed", _device_queue);

    _intermediate = popsift::sycl_common::malloc_devT<float>(
      _w * _h, __FILE__, __LINE__, "Intermediate allocation for octave failed", _device_queue);

    // Allocate all in one chunck (might be better to have it in multiple to have less chance of it failing but this is
    // propbs faster (might be insignificant))

    // std::stringstream data_msg; // could use std::format if c++20
    // data_msg << "Could not allocate all data levels as one segment of of size " << (_w * _h * _levels) / 1000 <<
    // "kB";

    _data_array_host[0] = popsift::sycl_common::malloc_devT<float>(
      _w * _h * _levels, __FILE__, __LINE__, "Could not allocate all data levels as one segment", _device_queue);

    // std::stringstream dog_msg; // could use std::format if c++20
    // dog_msg << "Could not allocate DoG levels as one segment of of size " << (_w * _h * (_levels - 1)) / 1000 <<
    // "kB";

    _dog_array_host[0] = popsift::sycl_common::malloc_devT<float>(
      _w * _h * (_levels - 1), __FILE__, __LINE__, "Could not allocate DoG levels as one segment", _device_queue);

    // Set the pointer positions for indexing
    for(int i = 1; i < _levels - 1; ++i)
    {
        _data_array_host[i] = _data_array_host[0] + (i * _w * _h);
        _dog_array_host[i] = _dog_array_host[0] + (i * _w * _h);
    }

    // Data has one more than dog hence out of loop
    _data_array_host[_levels - 1] = _data_array_host[0] + ((_levels - 1) * _w * _h);

    // Copy host arrays to device
    _data_array_write = _device_queue.memcpy(_data_array, _data_array_host, _levels * sizeof(float*));
    _dog_array_write = _device_queue.memcpy(_dog_array, _dog_array_host, (_levels - 1) * sizeof(float*));
}

// Assumes _levels can't change affter malloc have been done
void Octave::free_arrays()
{
    fprintf(stderr, "\nFREEING OCTAVE %d\n", _debug_octave_id);

    if(!_data_array)
    {
        fprintf(stderr, "\nData array is NULL at octave=%d\n", _debug_octave_id);
    }
    if(!_dog_array)
    {
        fprintf(stderr, "\nDOG array is NULL at octave=%d\n", _debug_octave_id);
    }

    if(!_intermediate)
    {
        fprintf(stderr, "\nINTERMEDIATE array is NULL at octave=%d\n", _debug_octave_id);
    }

    sycl::free(_data_array_host[0], _device_queue); // one large segment holding all levels
    sycl::free(_data_array, _device_queue);
    delete[] _data_array_host;

    sycl::free(_dog_array_host[0], _device_queue); // one large segment holding all levels
    sycl::free(_dog_array, _device_queue);
    delete[] _dog_array_host;

    sycl::free(_intermediate, _device_queue);
}

void Octave::resetDimensions(const Config& conf, int w, int h)
{
    if(w == _w && h == _h)
        return;

    if(w * h <= _max_w * _max_h)
    {
        // Smaller than current allocation hence we can reuse
        _w = w;
        _h = h;
        return;
    }

    // Larger than current segment -- need to free and allocate again

    free_arrays();

    // Only listens to changes when too small
    if(_levels - 3 != conf.levels) // conf.levels is searchable levels so there are 3 more octave levels
    {
        fprintf(stderr, "Levels have changed since initialization");
        // Could do this in all cases aswell IDK what is better

        // NOTE: Should mby have this run always to have config changes take effect. Currenly changes to filterGridsize
        // would not be updated unless also levels change
        alloc(conf, w, h, conf.levels + 3);
        return;
    }

    _max_w = _w = w;
    _max_h = _h = h;
    alloc_arrays();
}

/*************************************************************
 * OctaveBindless -- Bindless images array
 *************************************************************/

OctaveBindless::OctaveBindless(sycl::queue Q)
  : OctaveBase(Q)
{}

// Same as for Octave just that we need ro re-alloc for every change
void OctaveBindless::resetDimensions(const Config& conf, int w, int h)
{
    if(w == _w && h == _h)
        return;

    // Any change in dimensions requires re-alloc
    free_arrays();

    // Not needed as we reset the whole pyramid if the config changes
    if(_levels - 3 != conf.levels) // conf.levels is searchable levels so there are 3 more octave levels
    {
        fprintf(stderr, "Levels have changed since initialization");
        alloc(conf, w, h, conf.levels + 3);
        return;
    }

    _max_w = _w = w;
    _max_h = _h = h;
    alloc_arrays();
}

void OctaveBindless::alloc_arrays()
{
    const size_t w = static_cast<size_t>(_w);
    const size_t h = static_cast<size_t>(_h);

    // Descriptors
    _intermediate_desc = (syclexp::image_descriptor({w, h}, 1, sycl::image_channel_type::fp32));
    _data_desc =
      (syclexp::image_descriptor({w, h}, 1, sycl::image_channel_type::fp32, syclexp::image_type::array, 1, _levels));
    _dog_desc = (syclexp::image_descriptor(
      {w, h}, 1, sycl::image_channel_type::fp32, syclexp::image_type::array, 1, _levels - 1));

    // Not sure if using sampled is worth it or if it's better to just use unsampled
    // But at that point I'm not sure if there is any value in using bindless for Octave
    syclexp::bindless_image_sampler octave_sampler(sycl::addressing_mode::clamp_to_edge,
                                                   sycl::coordinate_normalization_mode::unnormalized,
                                                   sycl::filtering_mode::nearest);
    try
    {
        // Actual data objects
        _intermediate_mem = syclexp::alloc_image_mem(_intermediate_desc, _device_queue);
        _data_mem = syclexp::alloc_image_mem(_data_desc, _device_queue);
        _dog_mem = syclexp::alloc_image_mem(_dog_desc, _device_queue);

        // Handles to underlying data ^
        // sampled (not sure if they are beneficial)
        _intermediate_handle_read = popsift::sycl_bindless::create_sampled_image(
          _intermediate_mem, octave_sampler, _intermediate_desc, _device_queue);
        _data_handle_read =
          popsift::sycl_bindless::create_sampled_image(_data_mem, octave_sampler, _data_desc, _device_queue);
        _dog_handle_read =
          popsift::sycl_bindless::create_sampled_image(_dog_mem, octave_sampler, _dog_desc, _device_queue);

        // Unsampled
        _intermediate_handle_write = syclexp::create_image(_intermediate_mem, _intermediate_desc, _device_queue);
        _data_handle_write = syclexp::create_image(_intermediate_mem, _data_desc, _device_queue);
        _dog_handle_write = syclexp::create_image(_intermediate_mem, _dog_desc, _device_queue);

        // Should mby have one per action above but if it don't get here it's gone wrong anyways...
        // Aka not enought memory so if we don't free correctly it should not matter much as it will terminate soon
        // anyways
        _allocated = true;
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

    // Handles to data
}

void OctaveBindless::free_arrays()
{
    try
    {
        if(_allocated) // Single state for all (Might need one per to be safe)
        {
            // Destroy handles
            syclexp::destroy_image_handle(_intermediate_handle_read, _device_queue);
            syclexp::destroy_image_handle(_intermediate_handle_write, _device_queue);
            syclexp::destroy_image_handle(_data_handle_read, _device_queue);
            syclexp::destroy_image_handle(_data_handle_write, _device_queue);
            syclexp::destroy_image_handle(_dog_handle_read, _device_queue);
            syclexp::destroy_image_handle(_dog_handle_write, _device_queue);

            // Free underlying memory
            syclexp::free_image_mem(_intermediate_mem, syclexp::image_type::standard, _device_queue);
            syclexp::free_image_mem(_data_mem, syclexp::image_type::array, _device_queue);
            syclexp::free_image_mem(_dog_mem, syclexp::image_type::array, _device_queue);
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

}
