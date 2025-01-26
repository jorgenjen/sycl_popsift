#include "sycl_popsift/sift_octave.hpp"

namespace popsift {

Octave::Octave() {}

void Octave::free() {}

void Octave::alloc(const Config& conf, int width, int height, int levels, int gauss_group)
{
    _max_w = _w = width;
    _max_h = _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    // TODO: FIGURE out Replacements for these methods
    // most of them are related to textures in CUDA

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
