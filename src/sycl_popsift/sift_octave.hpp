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

// struct LinearTexture {
//     cudaSurfaceObject_t tex;
// };

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

    sycl::queue& _device_queue; // reference to  of device queue

    // Intermediate
    // TODO: change to only use a float* _intm_data (or whatever) as we only need one!!!
    // might just need one float _intm_array as next scale depends on prev hence only one is written to at once
    // float** _intm_array; // array pointing to array of float array
    // which each represents the different levels (blur)
    // in the octave

    float* _intermediate; // should not need an array

    float** _data_array;      // Gaussians stored _levels
    float** _data_array_host; // Just for memory management
    float** _dog_array;       // DoG stored _levels - 1
    float** _dog_array_host;  // Just for memory maangemrnt

    // cudaArray_t _data{};
    // cudaChannelFormatDesc _data_desc{};
    // cudaExtent _data_ext{};
    // cudaSurfaceObject_t _data_surf{};
    // cudaTextureObject_t _data_tex_point{};
    // LinearTexture _data_tex_linear{};

    // cudaArray_t _intm{};
    // cudaChannelFormatDesc _intm_desc{};
    // cudaExtent _intm_ext{};
    // cudaSurfaceObject_t _intm_surf{};
    // cudaTextureObject_t _intm_tex_point{};
    // LinearTexture _intm_tex_linear{};
    //
    // cudaArray_t _dog_3d{};
    // cudaChannelFormatDesc _dog_3d_desc{};
    // cudaExtent _dog_3d_ext{};
    // cudaSurfaceObject_t _dog_3d_surf{};
    // cudaTextureObject_t _dog_3d_tex_point{};
    //
    // // one CUDA stream per level
    // // consider whether some of them can be removed
    // cudaStream_t _stream{};
    // cudaEvent_t _scale_done{};
    // cudaEvent_t _extrema_done{};
    // cudaEvent_t _ori_done{};
    // cudaEvent_t _desc_done{};

    sycl::event _data_array_write;
    sycl::event _dog_array_write;

  public:
    std::vector<sycl::event> _level_complete_events;
    sycl::event _extrema_done_event;
    // Octave();
    Octave() = delete;
    Octave(sycl::queue& Q);
    ~Octave() { this->free_arrays(); }
    // ~Octave();

    void resetDimensions(const Config& conf, int w, int h);

    // DON'T know why this is set to uint32_t as both parameter passed and setter is int
    // hence no need for the extra numbers given by uint??
    // inline void debugSetOctave( uint32_t o ) { _debug_octave_id = o; }
    inline void debugSetOctave(int o) { _debug_octave_id = o; }

    inline int getLevels() const { return _levels; }
    inline int getWidth() const { return _w; }
    inline int getHeight() const { return _h; }

    inline float getWGridDivider() const { return _w_grid_divider; }
    inline float getHGridDivider() const { return _h_grid_divider; }

    inline float* getIntermediate() const { return _intermediate; }
    // inline float** getIntermediateArray() const { return _intm_array; }
    inline float** getDataArray() const { return _data_array; }
    inline float** getDataArrayHost() const { return _data_array_host; }
    inline sycl::event getDataArrayWriteEvent() const { return _data_array_write; }

    inline float** getDogArray() const { return _dog_array; }
    inline float** getDogArrayHost() const { return _dog_array_host; }
    inline sycl::event getDogArrayWriteEvent() const { return _dog_array_write; }

    // inline cudaStream_t getStream( ) {
    //     return _stream;
    // }
    // inline cudaEvent_t getEventScaleDone( ) {
    //     return _scale_done;
    // }
    // inline cudaEvent_t getEventExtremaDone( ) {
    //     return _extrema_done;
    // }
    // inline cudaEvent_t getEventOriDone( ) {
    //     return _ori_done;
    // }
    // inline cudaEvent_t getEventDescDone( ) {
    //     return _desc_done;
    // }
    //
    // inline LinearTexture getIntermDataTexLinear( ) {
    //     return _intm_tex_linear;
    // }
    // inline cudaTextureObject_t getIntermDataTexPoint( ) const {
    //     return _intm_tex_point;
    // }
    // inline LinearTexture getDataTexLinear( ) {
    //     return _data_tex_linear;
    // }
    // inline cudaTextureObject_t getDataTexPoint( ) const {
    //     return _data_tex_point;
    // }
    // inline cudaSurfaceObject_t getDataSurface( ) const {
    //     return _data_surf;
    // }
    // inline cudaSurfaceObject_t getIntermediateSurface( ) const {
    //     return _intm_surf;
    // }
    //
    // inline cudaSurfaceObject_t& getDogSurface( ) {
    //     return _dog_3d_surf;
    // }
    // inline cudaTextureObject_t& getDogTexturePoint( ) {
    //     return _dog_3d_tex_point;
    // }

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
    void alloc_arrays();
    void free_arrays();

    // void alloc_data_planes();
    // void alloc_data_tex();
    // void alloc_interm_array();
    // void alloc_interm_tex();
    // void alloc_dog_array();
    // void alloc_dog_tex();
    // void alloc_streams();
    // void alloc_events();
    //
    // void free_events();
    // void free_streams();
    // void free_dog_tex();
    // void free_dog_array();
    // void free_interm_tex();
    // void free_interm_array();
    // void free_data_tex();
    // void free_data_planes();
};

} // namespace popsift
