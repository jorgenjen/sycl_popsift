/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// #include "features.h"
// #include "sift_constants.h"
#include "sycl_popsift/features.hpp"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/s_image.hpp"
#include "sycl_popsift/sift_constants.hpp"
#include "sycl_popsift/sift_extremum.h"
#include "sycl_popsift/sift_octave.hpp"

#include <iostream>
#include <vector>

namespace popsift {

struct ExtremaCounters
{
    /* The number of extrema found per octave */
    int ext_ct[MAX_OCTAVES];
    /* The number of orientation found per octave */
    int ori_ct[MAX_OCTAVES];

    /* Exclusive prefix sum of ext_ct */
    int ext_ps[MAX_OCTAVES]; // TODO: look into removing this one
    /* Exclusive prefix sum of ori_ct */
    int ori_ps[MAX_OCTAVES];

    int ext_total;
    int ori_total;
};

struct ExtremaBuffers
{
    Descriptor* desc;
    int ext_allocated;
    int ori_allocated;
};

struct DevBuffers
{
    InitialExtremum* i_ext_dat[MAX_OCTAVES];
    int* i_ext_off[MAX_OCTAVES];
    int* feat_to_ext_map;
    Extremum* extrema;
    Feature* features;
};

// extern thread_local ExtremaCounters hct;
// extern __device__ ExtremaCounters dct;
// extern thread_local ExtremaBuffers hbuf;
// extern __device__ ExtremaBuffers dbuf;
// extern thread_local ExtremaBuffers dbuf_shadow;  // just for managing
// memories extern __device__ DevBuffers dobuf; extern thread_local DevBuffers
// dobuf_shadow;  // just for managing memories

class Pyramid
{
    int _num_octaves;
    int _levels;
    // Octave* _octaves;
    std::vector<Octave> _octaves;
    int _gauss_group;

    /* initial blur variables are used for Gauss table computation,
     * not needed on device */
    bool _assume_initial_blur;
    float _initial_blur;

    /* used to implement a global barrier per octave */
    int* _d_extrema_num_blocks;

    sycl::queue _device_queue;
    popsift::GaussInfo* _d_gauss;  // copy of same pointer as PopSift's _d_gauss and it deals with freeing it
    popsift::ConstInfo* _d_consts; // copy of same pointer as Popsift's _d_consts and it deals with freeing it

    // Global memory not supported by sycl
    ExtremaCounters _hct;  // host
    ExtremaCounters* _dct; // device

    ExtremaBuffers* _dbuf;       // device
    ExtremaBuffers _dbuf_host{}; // for memory management
    ExtremaBuffers _hbuf{};      // keeps host allocations (so fully host struct)
    sycl::event _dbuf_write;

    DevBuffers* _dobuf;       // device
    DevBuffers _dobuf_host{}; // needed for memory management
    sycl::event _dobuf_write;

    sycl::event _zero_dct;
    sycl::event _zero_extrema_num_blocks;

    /* the download of converted descriptors should be asynchronous */
    // cudaStream_t _download_stream;

  public:
    enum GaussTableChoice
    {
        Interpolated_FromPrevious,
        NotInterpolated_FromPrevious,
    };

  public:
    Pyramid(const Config& config,
            int w,
            int h,
            sycl::queue Q,
            popsift::GaussInfo* d_gauss,
            popsift::ConstInfo* d_consts,
            popsift::ConstInfo& h_consts);

    ~Pyramid();

    void resetDimensions(const Config& conf, int width, int height);

    /** step 1: load image and build pyramid */
    // std::vector<sycl::event> step1(const Config& conf, Image* img, sycl::event d_gauss_wirte, sycl::event
    // img_transfer);
    void step1(const Config& conf, Image* img, sycl::event d_gauss_wirte, sycl::event img_transfer);

    /** step 2: find extrema, orientations and descriptor */
    // void step2(const Config& conf, std::vector<sycl::event> dependencies, sycl::event d_consts_write);
    void step2(const Config& conf, sycl::event d_consts_write);

    /** step 3: download descriptors */
    FeaturesHost* get_descriptors(const Config& conf);

    /** step 3 (alternative): make copy of descriptors on device side */
    FeaturesDev* clone_device_descriptors(const Config& conf);

    void download_and_save_array(const char* basename);

    // void save_descriptors(const Config &conf, FeaturesHost *features,
    // const char *basename);

    inline int getNumOctaves() const { return _num_octaves; }
    inline int getNumLevels() const { return _levels; }

    inline Octave& getOctave(const int o) { return _octaves[o]; }

  private:
    // sycl::event horiz_from_input_image(const Config& conf, Image* base, sycl::event d_gauss_write);
    sycl::event horiz_from_input_image(const Config& conf,
                                       Image* base,
                                       sycl::event d_gauss_write,
                                       sycl::event img_write);

    inline sycl::event downscale_from_prev_octave(int octave);

    sycl::event horiz_from_prev_level_basic(int octave, int level);
    void horiz_from_prev_level_pairs(int octave, int level); // Not implemented as of now
    inline sycl::event horiz_from_prev_level(int octave, int level, GaussTableChoice useInterpolatedGauss);
    sycl::event vert_from_interm_basic(int octave, int level, sycl::event intm_write);
    // void vert_from_interm_pairs(int octave, int level, cudaStream_t stream);
    sycl::event vert_from_interm(int octave, int level, GaussTableChoice useInterpolatedGauss, sycl::event intm_write);

    // inline void dogs_from_blurred(int octave, int max_level, sycl::event octave_complete);
    sycl::event dogs_from_blurred(int octave, int max_level, sycl::event octave_complete);

    void reset_extrema_mgmt();
    void build_pyramid(const Config& conf, Image* base, sycl::event d_gauss_write, sycl::event img_transfer);
    void find_extrema(const Config& conf, sycl::event d_consts_write);
    void reallocExtrema(int numExtrema);

    int extrema_filter_grid(const Config& conf,
                            int ext_total); // called at head of orientation
    void orientation(const Config& conf);

    void descriptors(const Config& conf);

    inline void start_ext_desc_loop(const int octave, Octave& oct_obj, bool use_sub_group);

    sycl::event readDescCountersFromDevice();
    // void readDescCountersFromDevice(cudaStream_t s);
    void writeDescCountersToDevice();
    // void writeDescCountersToDevice(cudaStream_t s);
    int* getNumberOfBlocks(int octave);
    // void writeDescriptor(const Config &conf, std::ostream &ostr,
    //                      FeaturesHost *features, bool really,
    //                      bool with_orientation);

    void clone_device_descriptors_sub(const Config& conf, FeaturesDev* features);
};

} // namespace popsift
