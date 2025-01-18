/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// #include "../common/debug_macros.hpp"
#include "sift_conf.hpp"

#include <iostream>

using namespace std;

namespace popsift
{

Config::Config( )
    : _upscale_factor( 1.0f )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , _edge_limit( 10.0f )
    , _threshold( 0.04 ) // ( 10.0f / 256.0f )
    // , _gauss_mode( getGaussModeDefault() )
    , _sift_mode( Config::RefineInOctave )
    , _log_mode( Config::None )
    , _desc_mode( Config::Loop )
    , _grid_filter_mode( Config::RandomScale )
    , verbose( false )
    // , _max_extrema( 20000 ) // Uncommented in popsift aswell not done by Jørgen Jensovld :D
    , _max_extrema( 100000 )
    , _filter_max_extrema( -1 )
    , _filter_grid_size( 2 )
    , _assume_initial_blur( true )
    , _initial_blur( 0.5f )
    // , _normalization_mode( getNormModeDefault() )
    , _normalization_multiplier( 0 )
    , _print_gauss_tables( false )
{
    int            currentDev;
    // cudaDeviceProp currentProp;
    // cudaError_t    err;

    // err = cudaGetDevice( &currentDev );
    // POP_CUDA_FATAL_TEST( err, "Could not get current device ID" );
    //
    // err = cudaGetDeviceProperties( &currentProp, currentDev );
    // POP_CUDA_FATAL_TEST( err, "Could not get current device properties" );
}
}; // namespace popsift

