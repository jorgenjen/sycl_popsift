#include "sycl_popsift/sift_pyramid.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/s_image.hpp" // not sure if needed to include here aswell clean up #includes at some point
#include "sycl_popsift/sift_constants.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

namespace popsift {

Pyramid::Pyramid(const Config& config,
                 int width,
                 int height,
                 sycl::queue Q,
                 popsift::GaussInfo* d_gauss,
                 popsift::ConstInfo* d_consts,
                 popsift::ConstInfo& h_consts)
  : _num_octaves(config.octaves)
  , _levels(config.levels + 3)
  , _assume_initial_blur(config.hasInitialBlur())
  , _initial_blur(config.getInitialBlur())
  , _device_queue(Q)
  , _d_gauss(d_gauss)
  , _d_consts(d_consts)
{
    // _octaves = new Octave[_num_octaves];
    // Could not find a way to use C array so using vector

    fprintf(stderr, "Before emplace back and reserve\n");
    _octaves.reserve(_num_octaves);
    for(int i = 0; i < _num_octaves; ++i)
    {
        // _octaves.emplace_back(_device_queue.get_context(), _device_queue.get_device());
        _octaves.emplace_back(_device_queue);
    }
    fprintf(stderr, "After emplace back and reserve\n");

    int w = width;
    int h = height;
    for(int o = 0; o < _num_octaves; o++)
    {
        _octaves[o].debugSetOctave(o);

        _octaves[o].alloc(config, w, h, _levels);
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }

    // memset(&_hbuf, 0, sizeof(ExtremaBuffers)); // TODO: Do I need to and can you for stack memory?

    // Seems to be used alot but need to sync with dct when it is set to use I think so should
    // not be needed to zero it out
    // memset(&_hct, 0, sizeof(ExtremaCounters)); // might have a puprpose on the host side

    // NOTE: Not sure if we want to have the custom error message for each malloc not sure if it gives any usefull
    // information and it might slightly slow us down...

    _dct = popsift::sycl_common::malloc_devT<ExtremaCounters>(
      1, __FILE__, __LINE__, "Device Extrema counter allocation failed", _device_queue);
    // _device_queue.memset(_dct, 0, sizeof(ExtremaCounters)); // Should be a dependency for extrema?

    _d_extrema_num_blocks = popsift::sycl_common::malloc_devT<int>(
      _num_octaves, __FILE__, __LINE__, "Global octave barrier allocation failed", _device_queue);
    // _device_queue.memset(_d_extrema_num_blocks, 0, sizeof(int) * _num_octaves); // done in reset_extrema_mgmt
    // function that is called before first extema And results of this and dct copy are dependency for that kernel
    // launch

    int size = _num_octaves * h_consts.max_extrema; // h_consts.max_extrema is 100 000 by default
    _dobuf = popsift::sycl_common::malloc_devT<DevBuffers>(
      1, __FILE__, __LINE__, "Allocating device DevBuffers struct failed", _device_queue);

    // For 7 octaves case the total memory useage for this array is 196MB
    _dobuf_host.i_ext_dat[0] = popsift::sycl_common::malloc_devT<InitialExtremum>(
      size, __FILE__, __LINE__, "Device InitialExtremum array allocation failed", _device_queue);

    // For 7 octaves case the total memory useage for this array is 2.8MB
    _dobuf_host.i_ext_off[0] = popsift::sycl_common::malloc_devT<int>(
      size, __FILE__, __LINE__, "Device extremum offset array allocation failed", _device_queue);

    // All octaves in one contigous memory segment that has 100k each in default case
    // loop sets poitners to appropriate 0 positions per octave
    for(int o = 1; o < _num_octaves; o++)
    {
        _dobuf_host.i_ext_dat[o] = _dobuf_host.i_ext_dat[0] + (o * h_consts.max_extrema);
        _dobuf_host.i_ext_off[o] = _dobuf_host.i_ext_off[0] + (o * h_consts.max_extrema);
    }
    // set remaining to nullptr
    for(int o = _num_octaves; o < MAX_OCTAVES; o++)
    {
        _dobuf_host.i_ext_dat[o] = nullptr;
        _dobuf_host.i_ext_off[o] = nullptr;
    }

    size = h_consts.max_extrema; // setting to 100 000 in default case octave num invariant
    _dobuf_host.extrema = popsift::sycl_common::malloc_devT<Extremum>(
      size, __FILE__, __LINE__, "Device Extremum array allocation failed", _device_queue);
    _dobuf_host.features = popsift::sycl_common::malloc_devT<Feature>(
      size, __FILE__, __LINE__, "Device Feature array allocation failed", _device_queue);

    _dbuf = popsift::sycl_common::malloc_devT<ExtremaBuffers>(
      1, __FILE__, __LINE__, "Device ExtremaBuffer struct allocation failed", _device_queue);

    _hbuf.ext_allocated = size;
    _dbuf_host.ext_allocated = size;

    size = std::max(2 * h_consts.max_extrema, h_consts.max_orientations);
    _hbuf.desc = popsift::sycl_common::malloc_hostT<Descriptor>(
      size, __FILE__, __LINE__, "Host Descriptor array allocation failed", _device_queue);
    _dbuf_host.desc = popsift::sycl_common::malloc_devT<Descriptor>(
      size, __FILE__, __LINE__, "Device Descriptor array allocation failed", _device_queue);

    _dobuf_host.feat_to_ext_map = popsift::sycl_common::malloc_devT<int>(
      size, __FILE__, __LINE__, "Device feat_to_ext_map allocation failed", _device_queue);

    _hbuf.ori_allocated = size;
    _dbuf_host.ori_allocated = size;

    // NOTE: Consider using sycl::malloc_host for _dobuf as it's fairly large (344 bytes) and could take advantage from
    // pined memory
    // Probs good to move one up so that the transfer can start as early as possible while we malloc for the other
    _dobuf_write = _device_queue.memcpy(_dobuf, &_dobuf_host, sizeof(DevBuffers));
    _dbuf_write = _device_queue.memcpy(_dbuf, &_dbuf_host, sizeof(ExtremaBuffers));
}

// void Pyramid::reallocExtrema(int numExtrema)
// {
//     // Can happen as ext_allocated is same as max extrema per octave hence sum of octaves
//     // extrema could be higher than allocated (don't like that we need to sync up (blocking) after all extrema
//     compute)
//     // to do this. As we could otherwise just continue for orientation having only it's octave doing the extrema as
//     // dependency
//     if(numExtrema > hbuf.ext_allocated)
//     {
//         // Makes adds 1024 to size and removes all set bits that is below 1024 position in binary resulting in the
//         // segment being a multiple of 1024 (Probs yields better performance)
//         numExtrema = ((numExtrema + 1024) & (~(1024 - 1)));
//         cudaFree(dobuf_shadow.extrema);
//         cudaFree(dobuf_shadow.features);
//
//         int sz = numExtrema;
//         dobuf_shadow.extrema = popsift::cuda::malloc_devT<Extremum>(sz, __FILE__, __LINE__);
//         dobuf_shadow.features = popsift::cuda::malloc_devT<Feature>(sz, __FILE__, __LINE__);
//         hbuf.ext_allocated = sz;
//         dbuf_shadow.ext_allocated = sz;
//
//         numExtrema *= 2;
//         if(numExtrema > hbuf.ori_allocated)
//         {
//             cudaFreeHost(hbuf.desc);
//             cudaFree(dbuf_shadow.desc);
//             cudaFree(dobuf_shadow.feat_to_ext_map);
//
//             sz = numExtrema;
//             hbuf.desc = popsift::cuda::malloc_hstT<Descriptor>(sz, __FILE__, __LINE__);
//             dbuf_shadow.desc = popsift::cuda::malloc_devT<Descriptor>(sz, __FILE__, __LINE__);
//             dobuf_shadow.feat_to_ext_map = popsift::cuda::malloc_devT<int>(sz, __FILE__, __LINE__);
//             hbuf.ori_allocated = sz;
//             dbuf_shadow.ori_allocated = sz;
//         }
//
//         cudaMemcpyToSymbol(dbuf, &dbuf_shadow, sizeof(ExtremaBuffers), 0, cudaMemcpyHostToDevice);
//         cudaMemcpyToSymbol(dobuf, &dobuf_shadow, sizeof(DevBuffers), 0, cudaMemcpyHostToDevice);
//     }
// }

Pyramid::~Pyramid()
{
    // Octaves stored in vector so they will be destroyed/deleted by this object being destroyed
    fprintf(stderr, "Destroying the Pyramid!\n");
    sycl::free(_d_extrema_num_blocks, _device_queue);
    sycl::free(_dct, _device_queue);

    sycl::free(_dobuf_host.i_ext_dat[0], _device_queue);
    sycl::free(_dobuf_host.i_ext_off[0], _device_queue);
    sycl::free(_dobuf_host.features, _device_queue);
    sycl::free(_dobuf_host.extrema, _device_queue);
    sycl::free(_dobuf_host.feat_to_ext_map, _device_queue);

    sycl::free(_dbuf_host.desc, _device_queue);
    sycl::free(_hbuf.desc, _device_queue);

    // sycl::free(_dobuf_host, _device_queue); // No need as it's a struct not malloced
}

void Pyramid::step1(const Config& conf, popsift::Image* img, sycl::event d_gauss_write, sycl::event img_transfer)
{
    // TODO: Implement the reset -- far down the line need to find extrema first
    // reset_extrema_mgmt();

    // reset_extrema_mgmt(); // Required for first run aswell
    build_pyramid(conf, img, d_gauss_write, img_transfer);
}

// could probably pass the dependencies as a reference...
// void Pyramid::step2(const Config& conf, std::vector<sycl::event> dependencies, sycl::event d_consts_write)
void Pyramid::step2(const Config& conf, sycl::event d_consts_write)
{
    // TODO: Ensure no waits before this point

    // Was in step1 before build_pyramid before
    // Moved here as nothing in step1 requies this to be done and nothing blocks before this so it will be scheduled
    // quite quickly so should not add wait time but I mght be wrong (profile)
    reset_extrema_mgmt(); // Required for first run aswell

    // find_extrema(conf, dependencies, d_consts_write);
    find_extrema(conf, d_consts_write);

    orientation(conf);
    //
    // descriptors( conf );
}

void Pyramid::reset_extrema_mgmt()
{
    // memset(&_hct, 0, sizeof(ExtremaCounters)); // TODO: Figure out if I need this

    _zero_dct = _device_queue.memset(_dct, 0, sizeof(popsift::ExtremaCounters));
    _zero_extrema_num_blocks = _device_queue.memset(_d_extrema_num_blocks, 0, sizeof(int) * _num_octaves);
}

void Pyramid::resetDimensions(const Config& conf, int width, int height)
{
    int w = width;
    int h = height;

    for(int o = 0; o < _num_octaves; o++)
    {
        _octaves[o].resetDimensions(conf, w, h);
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }
}

// Fine to use on device memory as it is just pointer arithmetic
int* Pyramid::getNumberOfBlocks(int octave) { return &_d_extrema_num_blocks[octave]; }

sycl::event Pyramid::readDescCountersFromDevice() { return _device_queue.memcpy(&_hct, _dct, sizeof(ExtremaCounters)); }

} // namespace popsift
