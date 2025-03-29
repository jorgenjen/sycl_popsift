#include "sycl_popsift/sift_octave.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/debug_macros.hpp"

#include <sstream>

namespace popsift {

Octave::Octave(sycl::context ctx, sycl::device dev)
  : _device_queue(sycl::queue(ctx, dev))
{}

// Octave::~Octave() { free_arrays(); }

void Octave::alloc_arrays()
{
    try
    {
        // _intm_array = sycl::malloc_device<float*>(_levels, _device_queue);
        // _data_array = sycl::malloc_device<float*>(_levels, _device_queue);
        // _dog_array = sycl::malloc_device<float*>(_levels - 1, _device_queue);
        // _intermediate = sycl::malloc_device<float>(_w * _h, _device_queue);

        // NOTE: Not sure if I want these to be shared or find a way to only have them on host look deeper into it
        _data_array = sycl::malloc_shared<float*>(_levels, _device_queue);
        _dog_array = sycl::malloc_shared<float*>(_levels - 1, _device_queue);
        _intermediate = sycl::malloc_device<float>(_w * _h, _device_queue);
        // if(!_intm_array || !_data_array)
        if(!_data_array || !_intermediate) // should be caught by catch
        {
            POP_FATAL("Octave memory allocation failed");
        }
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << "Octave memory allocation failed" << e.what();
        POP_FATAL(ss.str());
        // std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }

    // fprintf(stderr, "After first try block\n");
    try
    {
        // Allocate the _levels in the octave
        for(int i = 0; i < _levels - 1; ++i)
        {
            _data_array[i] = sycl::malloc_device<float>(_w * _h, _device_queue);
            _dog_array[i] = sycl::malloc_device<float>(_w * _h, _device_queue);

            if(!_data_array[i] || !_dog_array[i])
            {
                POP_FATAL("Octave memory allocation failed");
            }
        }

        // Data has one more than dog hence out of loop
        _data_array[_levels - 1] = sycl::malloc_device<float>(_w * _h, _device_queue);
        if(!_data_array[_levels - 1])
        {
            POP_FATAL("Octave memory allocation failed");
        }
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << "Octave memory allocation failed" << e.what();
        POP_FATAL(ss.str());
        // std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }
    // fprintf(stderr, "After second try block\n");
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

    for(int i = 0; i < _levels - 1; ++i)
    {
        sycl::free(_data_array[i], _device_queue);
        sycl::free(_dog_array[i], _device_queue);
    }
    sycl::free(_data_array[_levels - 1], _device_queue); // has one more than DoG's

    sycl::free(_data_array, _device_queue);
    sycl::free(_dog_array, _device_queue);
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
    // fprintf(stderr, "Before alloc_arrays \n");
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
