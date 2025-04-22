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

#include <sycl/sycl.hpp>
// #include "sift_constants.h"
// #include "sift_extremum.h"

#include <iostream>
#include <vector>

namespace popsift {

class OctaveBase
{
  protected:
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

    sycl::event* _level_complete_events{nullptr};
    sycl::event _dog_done_event;
    sycl::event _extrema_done_event;

    friend class Pyramid; // Access to everything (can't restrict a friend :D)

    OctaveBase() = delete;
    OctaveBase(sycl::queue Q);

    void alloc(const Config& conf, int width, int height, int levels);

    virtual void resetDimensions(const Config& conf, int w, int h) = 0;
    virtual void alloc_arrays() = 0;
    virtual void free_arrays() = 0;

    inline void setLevelCompleteEvent(const int level, sycl::event e) const { _level_complete_events[level] = e; }
    /**
     * debug:
     * download a level and write to disk
     */
    void download_and_save_array(const char* basename, int octave); // need to implement

  public:
    // Assumes correct usage and does not check bounds
    inline sycl::event getLevelCompleteEvent(const int level) const { return _level_complete_events[level]; }
    inline sycl::event getDogDoneEvent() const { return _dog_done_event; }
    inline sycl::event getExtremaDoneEvent() const { return _extrema_done_event; }
};

class Octave : public OctaveBase
{
    float* _intermediate; // should not need an array

    float** _data_array;      // Gaussians stored _levels
    float** _data_array_host; // Just for memory management
    float** _dog_array;       // DoG stored _levels - 1
    float** _dog_array_host;  // Just for memory maangemrnt

    sycl::event _data_array_write;
    sycl::event _dog_array_write;

  public:
    // Octave();
    Octave() = delete;
    Octave(sycl::queue Q);
    ~Octave()
    {
        fprintf(stderr, "\n\tDESTROY OCTAVE\n");
        this->free_arrays();
        delete[] _level_complete_events;
    }

    void resetDimensions(const Config& conf, int w, int h) override;

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

  private:
    void alloc_arrays() override;
    void free_arrays() override;
};

class OctaveBindless : public OctaveBase
{
    OctaveBindless() = delete;
    OctaveBindless(sycl::queue);
    ~OctaveBindless()
    {
        fprintf(stderr, "\n\tDESTROY OCTAVEBINDLESS\n");
        this->free_arrays();
        delete[] _level_complete_events;
    }

    void resetDimensions(const Config& conf, int w, int h) override;
    // void alloc(const Config& conf, int width, int height, int levels) override;

    void alloc_arrays() override;
    void free_arrays() override;

    // Not sure if we want sampled or not (not sure how Out-of-boudswrites are handled)
    sycl::ext::oneapi::experimental::image_descriptor _intermediate_desc;               // Descriptor for image and
    sycl::ext::oneapi::experimental::image_mem_handle _intermediate_mem;                // Underlying meory
    sycl::ext::oneapi::experimental::sampled_image_handle _intermediate_handle_read;    // read only handle
    sycl::ext::oneapi::experimental::unsampled_image_handle _intermediate_handle_write; // write only handle

    sycl::ext::oneapi::experimental::image_descriptor _data_desc;               // Descriptor for image and handle
    sycl::ext::oneapi::experimental::image_mem_handle _data_mem;                // Underlying memory
    sycl::ext::oneapi::experimental::sampled_image_handle _data_handle_read;    // read only handle
    sycl::ext::oneapi::experimental::unsampled_image_handle _data_handle_write; // write only handle

    sycl::ext::oneapi::experimental::image_descriptor _dog_desc;               // Descriptor for image and handle
    sycl::ext::oneapi::experimental::image_mem_handle _dog_mem;                // Underlying meory
    sycl::ext::oneapi::experimental::sampled_image_handle _dog_handle_read;    // read only handle
    sycl::ext::oneapi::experimental::unsampled_image_handle _dog_handle_write; // write only handle

    bool _allocated{false};
};

} // namespace popsift
