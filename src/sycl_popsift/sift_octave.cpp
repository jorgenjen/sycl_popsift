#include "sycl_popsift/sift_octave.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/malloc_devt.hpp"

#include <sstream>

namespace popsift {

Octave::Octave(sycl::queue& Q)
  : _device_queue(Q)
{}

// Octave::~Octave() { free_arrays(); }

void Octave::alloc_arrays()
{
    // Could use shared for these ones aswell probs same end result and cleaner code

    _data_array_host =
      popsift::common::new_hostT<float*>(_levels, __FILE__, __LINE__, "Host allocation for data array failed");
    _dog_array_host =
      popsift::common::new_hostT<float*>(_levels - 1, __FILE__, __LINE__, "Host allocation for DoG array failed");

    _data_array = popsift::common_sycl::malloc_devT<float*>(
      _levels, __FILE__, __LINE__, "Device allocation for data array failed", _device_queue);

    _dog_array = popsift::common_sycl::malloc_devT<float*>(
      _levels - 1, __FILE__, __LINE__, "Device allocation for DoG array failed", _device_queue);

    _intermediate = popsift::common_sycl::malloc_devT<float>(
      _w * _h, __FILE__, __LINE__, "Intermediate allocation for octave failed", _device_queue);

    // Allocate all in one chunck (might be better to have it in multiple to have less chance of it failing but this is
    // propbs faster (might be insignificant))

    // std::stringstream data_msg; // could use std::forat if c++20
    // data_msg << "Could not allocate all data levels as one segment of of size " << (_w * _h * _levels) / 1000 <<
    // "kB";

    _data_array_host[0] = popsift::common_sycl::malloc_devT<float>(
      _w * _h * _levels, __FILE__, __LINE__, "Could not allocate all data levels as one segment", _device_queue);

    // std::stringstream dog_msg; // could use std::forat if c++20
    // dog_msg << "Could not allocate DoG levels as one segment of of size " << (_w * _h * (_levels - 1)) / 1000 <<
    // "kB";

    _dog_array_host[0] = popsift::common_sycl::malloc_devT<float>(
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

    // for(int i = 0; i < _levels - 1; ++i)
    // {
    //     sycl::free(_data_array[i], _device_queue);
    //     sycl::free(_dog_array[i], _device_queue);
    // }
    // sycl::free(_data_array[_levels - 1], _device_queue); // has one more than DoG's

    sycl::free(_data_array_host[0], _device_queue); // one large segment holding all levels
    sycl::free(_data_array, _device_queue);
    delete[] _data_array_host;

    sycl::free(_dog_array_host[0], _device_queue); // one large segment holding all levels
    sycl::free(_dog_array, _device_queue);
    delete[] _dog_array_host;

    sycl::free(_intermediate, _device_queue);
}

// void Octave::alloc(const Config& conf, int width, int height, int levels, int gauss_group, sycl::queue& Q)
void Octave::alloc(const Config& conf, int width, int height, int levels)
{
    _max_w = _w = width;
    _max_h = _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    _level_complete_events.reserve(levels);

    // TODO: FIGURE out Replacements for these methods
    // most of them are related to textures in CUDA

    // could store them all in one malloc (single float array) but might be less readable
    // and don't think there is much performance penalty from doing it this way...
    alloc_arrays();

    // alloc_data_planes();
    // alloc_data_tex();
    //
    // alloc_interm_array();
    // alloc_interm_tex();
    //
    // alloc_dog_array();
    // alloc_dog_tex();
    //
    // alloc_streams();
    // alloc_events();
}

void Octave::resetDimensions(const Config& conf, int w, int h)
{
    if(w == _w && h == _h)
        return;

    // This could result in worse performance than reallocating
    // as far as I understand it could lead to access starting
    // from a non cache-algned-address and hence end up using two cache lanes
    // instead ofo one evethout we access coaleced memory that is 128 multiple wide.
    //   "L1/TEX and L2 have 128B cache lines. Cache lines consist of 5 32B sectors.
    //   The tag lookup is at 128B granularity." From cuda forum
    // So might be better of freeing and reallocating to avoid this issue
    // or do some math check of the remainder to see if we hit cache aligned memory or not
    // As from my readng it seems like cudamalloc and cudafree (device malloc and free) are
    // quite expensive (much more so than cpu malloc and free) Should test this and make cases
    // where it ends up using two cache lines if  I can and compare free and malloc
    // vs using two cache lanes an interesting experiment would be to see how many frames
    // on that dimension woiuld need to be computed for it to be beneficial (if any)
    //     DISCLAIMER: I could have fully misunderstood cache lines with respect to memory
    //     segments and the only thing that matters is that you read coaleced memory but the
    //     said you could have non cache aligned address so tha's why  I assume this could
    //     be the case
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

    // free_dog_tex();
    // free_dog_array();
    //
    // free_interm_tex();
    // free_interm_array();
    //
    // free_data_tex();
    // free_data_planes();
    //
    // alloc_data_planes();
    // alloc_data_tex();
    //
    // alloc_interm_array();
    // alloc_interm_tex();
    //
    // alloc_dog_array();
    // alloc_dog_tex();
}
}
