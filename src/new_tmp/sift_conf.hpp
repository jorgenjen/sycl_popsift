/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <string>

#define MAX_OCTAVES   20
#define MAX_LEVELS    10

#ifdef _MSC_VER
#define DEPRECATED(func) __declspec(deprecated) func
#elif defined(__GNUC__) || defined(__clang__)
#define DEPRECATED(func) func __attribute__ ((deprecated))
#else
#endif

namespace popsift {

/**
 * @brief Struct containing the parameters that control the extraction algorithm
 */
struct Config
{
    Config();

    /**
     * @brief The way the gaussian mode is compute.
     *
     * Each setting allows to mimic and reproduce the behaviour of other Sift implementations.
     */
    enum GaussMode
    {
        VLFeat_Compute,
        VLFeat_Relative
    };

    /**
     * @brief General setting to reproduce the results of other Sift implementations.
     */
    enum SiftMode
    {
        /// refining an initial extremum stays in the same level of an octave
        RefineInLevel,
        /// refining an initial extremum can change level but stays in the same octave
        RefineInOctave,

        // PopSift = RefineInOctave, ///< Popsift implementation
        // VLFeat  = RefineInLevel,  ///< VLFeat implementation
        Default = RefineInOctave
    };

    /**
     * @brief The logging mode.
     */
    enum LogMode
    {
        None,
        All
    };

    /**
     * @brief Modes for descriptor extraction.
     */
    enum DescMode
    {
        /// scan horizontal, extract valid points
        Loop,
        /// scan horizontal, extract valid points, interpolate with tex engine
        ILoop,
        /// scan in rotated mode, round pixel address
        Grid,
        /// scan in rotated mode, interpolate with tex engine
        IGrid,
        /// variant of IGrid, no duplicate gradient fetching
        NoTile
    };

    /**
     * @brief Type of norm to use for matching.
     */
    enum NormMode
    {
        /// The L1-inspired norm, gives better matching results ("RootSift")
        RootSift,
        /// The L2-inspired norm, all descriptors on a hypersphere ("classic")
        Classic
    };

    /**
     * @brief Filtering strategy.
     * 
     * To reduce time used in descriptor extraction, some extrema can be filtered
     * immediately after finding them. It is possible to keep those with the largest
     * scale (LargestScaleFirst), smallest scale (SmallestScaleFirst), or a random
     * selection. Note that largest and smallest give a stable result, random does not.
     */
    enum GridFilterMode {
        /// keep a random selection
        RandomScale,
        /// keep those with the largest scale
        LargestScaleFirst,
        /// keep those with the smallest scale
        SmallestScaleFirst
    };

    /**
     * @brief Processing mode. 
     * 
     * Determines which data is kept in the Job data structure after processing, which one is downloaded to the host,
     * which one is invalidated.
     */
    enum ProcessingMode {
        ExtractingMode,
        MatchingMode
    };


    /// The number of octaves is chosen freely. If not specified,
    /// it is: log_2( min(x,y) ) - 3 - start_sampling
    int      octaves;

    /// The number of levels per octave. This is actually the
    /// number of inner DoG levels where we can search for
    /// feature points. The number of ...
    ///
    /// This is the non-augmented number of levels, meaning
    /// the this is not the number of gauss-filtered picture
    /// layers (which is levels+3), but the number of DoG
    /// layers in which we can search for extrema.
    int      levels;
    float    sigma;

    /// default edge_limit 16.0f from Celebrandil
    /// default edge_limit 10.0f from Bemap
    float    _edge_limit;


    /**
     * @brief The input image is stretched by 2^upscale_factor
     * before processing. The factor 1 is default.
     */
    inline float getUpscaleFactor( ) const {
        return _upscale_factor;
    }

    int getMaxExtrema( ) const {
        return _max_extrema;
    }


private:
    /// default threshold 0.0 default of vlFeat
    /// default threshold 5.0 / 256.0
    /// default threshold 15.0 / 256.0 - it seems our DoG is really small ???
    /// default threshold 5.0 from Celebrandil, not happening in our data
    /// default threshold 0.04 / (_levels-3.0) / 2.0f * 255
    ///                   from Bemap -> 1.69 (makes no sense)
    float    _threshold;

    /// determine the image format of the first octave
    /// relative to the input image's size (x,y) as follows:
    /// (x / 2^start_sampling, y / 2^start_sampling )
    float    _upscale_factor;

    /// default LogMode::None
    LogMode  _log_mode;

    /// default: DescMode::Loop
    DescMode    _desc_mode;

    /// default: RandomScale
    GridFilterMode _grid_filter_mode;

public:
    bool     verbose;

private:
    /// The number of initial extrema that can be discovered in an octave.
    /// This parameter changes memory requirements.
    int _max_extrema;

    /// The maximum number of extrema that are returned. There may be
    /// several descriptors for each extremum.
    int _filter_max_extrema;

    /// Used to achieve an approximation of _max_entrema
    /// Subdivide the image in this number of vertical and horizontal tiles,
    /// i.e. the grid is actually _grid_size X _grid_size tiles.
    /// default: 1
    int  _filter_grid_size;

    /// Modes are computation according to VLFeat or OpenCV,
    /// or fixed size. Default is VLFeat mode.
    // GaussMode _gauss_mode;

    /// Modes are PopSift, OpenCV and VLFeat.
    /// Default is currently identical to PopSift.
    SiftMode _sift_mode;

    /// VLFeat code assumes that an initial input image is partially blurred.
    /// This changes the blur computation for the very first level of the first
    /// octave, turning it into a special case.
    bool  _assume_initial_blur;
    float _initial_blur;

    /// OpenMVG requires a normalization named rootSift, the
    /// classical L2-inspired mode is also supported.
    // NormMode _normalization_mode;

    /// SIFT descriptors are normalized in a final step.
    /// The values of the descriptor can also be multiplied
    /// by a power of 2 if required.
    /// Specify the exponent.
    int _normalization_multiplier;

    /// Call the debug functions in gauss_filter.cu to print Gauss
    /// filter width and Gauss tables in use.
    bool _print_gauss_tables;
};

}; // namespace popsift

