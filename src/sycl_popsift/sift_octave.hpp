/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// don't think this is used and hence needed
// #include "sycl_popsift/s_image.hpp"
#include "sycl/queue.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
// #include "sift_constants.h"
// #include "sift_extremum.h"

#include <iostream>
#include <vector>

namespace popsift {

class Octave
{
    int _w{};
    int _h{};
    int _max_w{};
    int _max_h{};
    float _w_grid_divider{};
    float _h_grid_divider{};
    int _debug_octave_id{};
    int _levels{};
    int _gauss_group{};

    sycl::queue _device_queue;

    float* _intermediate; // should not need an array

    float** _data_array;      // Gaussians stored _levels
    float** _data_array_host; // Just for memory management
    float** _dog_array;       // DoG stored _levels - 1
    float** _dog_array_host;  // Just for memory maangemrnt

    sycl::event _data_array_write;
    sycl::event _dog_array_write;

  public:
    sycl::event* _level_complete_events;
    sycl::event _dog_done_event;
    sycl::event _extrema_done_event;

    // Octave();
    Octave() = delete;
    Octave(sycl::queue Q);
    ~Octave()
    {
        fprintf(stderr, "\n\tDESTROY OCTAVE\n");
        this->free_arrays();
    }

    void resetDimensions(const Config& conf, int w, int h);

    inline void debugSetOctave(int o) { _debug_octave_id = o; }

    inline int getLevels() const { return _levels; }
    inline int getWidth() const { return _w; }
    inline int getHeight() const { return _h; }

    inline float getWGridDivider() const { return _w_grid_divider; }
    inline float getHGridDivider() const { return _h_grid_divider; }

    inline float* getIntermediate() const { return _intermediate; }
    inline float** getDataArray() const { return _data_array; }
    inline float** getDataArrayHost() const { return _data_array_host; }

    inline sycl::event getDataArrayWriteEvent() const { return _data_array_write; }

    inline float** getDogArray() const { return _dog_array; }
    inline float** getDogArrayHost() const { return _dog_array_host; }
    inline sycl::event getDogArrayWriteEvent() const { return _dog_array_write; }

    /**
     * @brief Allocates all GPU memories for one octave.
     * @param conf
     * @param width in floats
     * @param height
     * @param levels
     * @param gauss_group
     */
    void alloc(const Config& conf, int width, int height, int levels);

    /**
     * debug:
     * download a level and write to disk
     */
    void download_and_save_array(const char* basename, int octave); // need to implement

  private:
    void alloc_bindless_arrays(); // For bindless_images array extension
    void alloc_arrays();
    void free_arrays();
};
} // namespace popsift
