#include "sycl_popsift/sift_octave.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/debug_macros.hpp"

namespace popsift {

Octave::Octave(sycl::queue& Q)
  : _device_queue(Q)
{}

Octave::~Octave()
{
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

void Octave::alloc(const Config& conf, int width, int height, int levels, int gauss_group, sycl::queue& Q)
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
    try
    {
        // _intm_array = sycl::malloc_device<float*>(levels, Q);
        _data_array = sycl::malloc_device<float*>(levels, Q);
        _dog_array = sycl::malloc_device<float*>(levels - 1, Q);
        _intermediate = sycl::malloc_device<float>(width * height, Q);

        // if(!_intm_array || !_data_array)
        if(!_data_array || !_intermediate)
        {
            POP_FATAL("Octave memory allocation failed");
        }
    }
    catch(const sycl::exception& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }
    try
    {
        // Allocate the levels in the octave
        for(int i = 0; i < levels - 1; ++i)
        {
            _data_array[i] = sycl::malloc_device<float>(width * height, Q);
            _dog_array[i] = sycl::malloc_device<float>(width * height, Q);

            if(!_data_array[i] || !_dog_array[i])
            {
                POP_FATAL("Octave memory allocation failed");
            }
        }

        // Data has one more than dog hence out of loop
        _data_array[levels - 1] = sycl::malloc_device<float>(width * height, Q);
        if(!_data_array[levels - 1])
        {
            POP_FATAL("Octave memory allocation failed");
        }
    }
    catch(const sycl::exception& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }

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
    {
        return;
    }

    _w = w;
    _h = h;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    if(_w > _max_w || _h > _max_h)
    {
        _max_w = std::max(_w, _max_w);
        _max_h = std::max(_h, _max_h);
    }

    // TODO: FInd replacements for these functions

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
