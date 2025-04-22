#include "sycl_popsift/sift_octave.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/debug_macros.hpp"

#include <sstream>

namespace popsift {

Octave::Octave(sycl::queue Q)
  : _device_queue(Q)
{}

void Octave::alloc(const Config& conf, int width, int height, int levels)
{
    _max_w = _w = width;
    _max_h = _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    _level_complete_events = new sycl::event[levels];

    alloc_arrays();
}

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

    // WARNING: Might have to add some form of verifcation that levels have no changed here
    // and deal with it appropriately
    if(_levels - 3 != conf.levels) // conf.levels is searchable levels so there are 3 more octave levels
    {
        fprintf(stderr, "Levels have changed since initialization");
        // Could do this in all cases aswell IDK what is better
        alloc(conf, w, h, conf.levels + 3);
        return;
    }

    _max_w = _w = w;
    _max_h = _h = h;
    alloc_arrays();
}

}
