/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
// #include "common/clamp.h"
#include "common/debug_macros.hpp"
// #include "s_solve.h" # Need this one later on
#include "sift_constants.hpp"
#include "sift_pyramid.hpp"
#include "sycl/nd_item.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"

// #include <cuda_runtime.h>
// #include <texture_fetch_functions.h>

#include <cstdio>
#include <vector>

namespace popsift {

class find_extrema_in_dog
{
    float** dog;
    int octave;
    int width;
    int height;
    const size_t max_level; // try int and see if it compiles
    int* number_of_blocks;
    const float w_grid_divider;
    const float h_grid_divider;
    const int grid_width;

    find_extrema_in_dog(float** dog,
                        int octave,
                        int width,
                        int height,
                        const size_t max_level,
                        int* number_of_blocks,
                        const float w_grid_divider,
                        const float h_grid_divider,
                        const int grid_width)
      : dog(dog)
      , octave(octave)
      , width(width)
      , height(height)
      , max_level(max_level)
      , number_of_blocks(number_of_blocks)
      , w_grid_divider(w_grid_divider)
      , h_grid_divider(h_grid_divider)
      , grid_width(grid_width)
    {}

    inline void operator()(sycl::nd_item<3> it) const
    {
        // code here yes mate !
        // bool indicator = find_extrema_in_dog_sub<sift_mode>(
        //   dog, octave, width, height, max_level, w_grid_divider, h_grid_divider, grid_width, ec);
    }
};

void Pyramid::find_extrema(const Config& conf, std::vector<sycl::event> dependencies)
{
    static const int HEIGHT = 4;

    for(int octave = 0; octave < _num_octaves; octave++)
    {
        Octave& oct_obj = _octaves[octave];

        // int* extrema_num_blocks = getNumberOfBlocks(octave); // not ready for this :C

        // dim3 block(32, HEIGHT);
        // dim3 grid;
        // grid.x = grid_divide(cols, block.x);
        // grid.y = grid_divide(rows, block.y);
        // grid.z = _levels - 3;

        // cudaStream_t oct_str = oct_obj.getStream();

        // int* num_blocks = extrema_num_blocks;

        int width = oct_obj.getWidth();
        int height = oct_obj.getHeight();

        sycl::range local{32, HEIGHT};
        sycl::range global{
          (size_t)grid_divide(width, local.get(0)), (size_t)grid_divide(height, local.get(1)), (size_t)_levels - 3};

        switch(conf.getSiftMode())
        {
            case Config::RefineInLevel:
                printf("RefineInLevel type VLfeat, NOT IMPLEMENTED AS OF NOW");
                // find_extrema_in_dog<HEIGHT, Config::RefineInLevel>
                //   <<<grid, block, 0, oct_str>>>(oct_obj.getDogTexturePoint(),
                //                                 octave,
                //                                 cols,
                //                                 rows,
                //                                 _levels - 1,
                //                                 num_blocks,
                //                                 grid.x * grid.y,
                //                                 oct_obj.getWGridDivider(),
                //                                 oct_obj.getHGridDivider(),
                //                                 conf.getFilterGridSize());
                // POP_SYNC_CHK;
                break;
            default:
                printf("RefineInOctave type popsift default\n");
                // _device_queue.parallel_for(sycl::nd_range{globa, local},
                //                            dependencies,
                //                            find_extrema_in_dog(oct_obj.getDogArray(),
                //                                                octave,
                //                                                width,
                //                                                height,
                //                                                _levels - 1,
                //                                                // num_blocks,
                //                                                global.get(0) * global.get(1),
                //                                                oct_obj.getWGridDivider(),
                //                                                oct_obj.getHGridDivider(),
                //                                                conf.getFilterGridSize()));
                break;
        }

        // cuda::event_record(oct_obj.getEventExtremaDone(), oct_str, __FILE__, __LINE__);
    }
}

} // namespace popsift
