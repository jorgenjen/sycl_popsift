#include "sycl_popsift/popsift.hpp"

#include "sycl/device.hpp"
#include "sycl/device_selector.hpp"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/features.hpp"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/sift_constants.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

// #include <sycl/ext/oneapi/device_info.hpp> // For Num compute units extension
#include <sycl/sycl.hpp>

#include <cmath> // ceilf
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
// using namespace std;
using std::cout;
using std::endl;
using std::max;
using std::min;

// #include <cuda_runtime_api.h>
//
// int get_max_warps_per_sm()
// {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     return prop.maxThreadsPerMultiProcessor / prop.warpSize;
// }

#define PRINT_MATRIX_OPTIONS 0

namespace syclexp = sycl::ext::oneapi::experimental;

#if PRINT_MATRIX_OPTIONS
std::string matrix_type_to_string(syclexp::matrix::matrix_type type)
{
    switch(type)
    {
        case syclexp::matrix::matrix_type::fp16: return "fp16";
        case syclexp::matrix::matrix_type::bf16: return "bf16";
        case syclexp::matrix::matrix_type::tf32: return "tf32";
        case syclexp::matrix::matrix_type::fp32: return "fp32";
        case syclexp::matrix::matrix_type::fp64: return "fp64";
        case syclexp::matrix::matrix_type::sint8: return "sint8";
        case syclexp::matrix::matrix_type::uint8: return "uint8";
        case syclexp::matrix::matrix_type::sint32: return "sint32";
        case syclexp::matrix::matrix_type::uint32: return "uint32";
        default: return "unknown";
    }
}
#endif

inline void PopSift::set_sg_per_cu()
{
// Decide Number of sub_groups per Compute Unit
// Not the best way of doing it and need's to be maintained. Can also be supplied in config as argument which then is
// used instead of this selection
// Should be replaced if possible to make this selection accurately at runtime
#ifdef CUDA_CC
    if constexpr(CUDA_CC == 86 || CUDA_CC == 89 || CUDA_CC == 30 || CUDA_CC == 35)
    {
        sg_per_cu = 48;
    }
    else if constexpr(CUDA_CC < 20)
    {
        sg_per_cu = 24;
    }
    else if constexpr(CUDA_CC < 35)
    {
        sg_per_cu = 32;
    }
    else
    {
        // Most common case
        sg_per_cu = 64;
    }
#endif

#ifdef HIP_ARCH

#define STRINGIFY(x) #x                  // Adds quotes around x
#define STRINGIFY_EXPAND(x) STRINGIFY(x) // Forces expansion of x first

    const char* gfx_arch_str = STRINGIFY_EXPAND(HIP_ARCH);
    const int gfx_version = std::atoi(gfx_arch_str + 3); // Remove gfx part of name (untested no access to amd card)

    if(gfx_version > 1000 && gfx_version < 1030)
    {
        sg_per_cu = 40; // RDNA1 has 40 wavefronts per CU the rest of modern has 32
    }
    else
    {
        sg_per_cu = 32;
    }

#endif

    // If it's intel GPU I've not found a good way and thye are split in to slices and seem to be more dynamic so not so
    // easy to figure out in that case would be better to use the config to set the value for the card you are using

    // Testing different values for cards can always be beneficial. If not set and it stays at -1 then persistent
    // threads will not be used
}

inline void PopSift::initQueue()
{
#if QUEUE_PROFILING
    sycl::property_list queue_proplist = sycl::property_list{sycl::property::queue::enable_profiling{}};
#else
    sycl::property_list queue_proplist = {};
#endif

#ifndef CPU_ONLY
    // should probably also have a compile time flag --experimental to enable this feature
    if constexpr(USE_BINDLESS_INPUT && USE_BINDLESS_ARRAY &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d>() &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_image_array>())
    {
        // Running with bindless image -- need to find gpu with that aspect (needed incase of multi gpu system)

        for(sycl::device dev : sycl::device::get_devices(sycl::info::device_type::gpu))
        {
            // Find GPU with the aspect (incase of multigpu system)
            if(dev.has(sycl::aspect::ext_oneapi_bindless_images) && dev.has(sycl::aspect::ext_oneapi_image_array) &&
               dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d))
            {
                std::cout << "Running on: " << _device_queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl
                          << "\t--> supports ext_oneapi_bindless_images: YES" << std::endl
                          << "\t--> supports ext_oneapi_image_array: YES" << std::endl;

                // _device_queue = sycl::queue(sycl::context{dev}, dev);
                _device_queue = sycl::queue(sycl::context{dev}, dev, queue_proplist);

                return; // We always select first gpu that had the aspect (might be a way to select the best one)
                        // but most systems will be single gpu anyways
            }
        }
        // Did not return hence we did not find a matching device to that was on the compiled system
        POP_FATAL("Could not find device with support for  ext_oneapi_bindless_images and ext_oneapi_image_array "
                  "Such a device was available at compile time... Please re-compile")
    }
    else if constexpr(USE_BINDLESS_INPUT && sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                      sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d>())
    {
        // In case it only supports bindless we can use it for upscaling still
        for(sycl::device dev : sycl::device::get_devices(sycl::info::device_type::gpu))
        {
            // Find GPU with the aspect (incase of multigpu system)
            if(dev.has(sycl::aspect::ext_oneapi_bindless_images) &&
               dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d))
            {
                std::cout << "Running on: " << _device_queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl
                          << "\t--> supports ext_oneapi_bindless_images: YES" << std::endl
                          << "\t--> supports ext_oneapi_image_array: "
                          << (dev.has(sycl::aspect::ext_oneapi_image_array)
                                ? "YES... But not in use due to being NO at compile time..."
                                : "NO")
                          << std::endl
                          << std::endl;

                _device_queue = sycl::queue(sycl::context{dev}, dev, queue_proplist);
                break; // We always select first gpu that had the aspect (might be a way to select the best one)
                       // but most systems will be single gpu anyways
            }
        }

        // Did not return hence we did not find a matching device to that was on the compiled system
        POP_FATAL("Could not find device with ext_oneapi_bindless_images support... Such a device was  available "
                  "at compile time... Please re-compile")
    }
    else
    {
        try
        {
            // Did not have bindless aspect during compile time so we just try to select any GPU
            // If there is no GPU it will throw exception and use CPU in catch

            sycl::device dev = sycl::device{sycl::gpu_selector_v};
            _device_queue = sycl::queue(sycl::context{dev}, dev, queue_proplist);
        }
        catch(sycl::exception const& ex)
        {
            cout << "No GPU found falling back to CPU... Exception thrown: " << ex.what() << endl;

            // Could use defualt selector but not sure how would handle fpga... hence cpu selector
            sycl::device dev = sycl::device{sycl::cpu_selector_v};
            _device_queue = sycl::queue(sycl::context{dev}, dev);
            // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
        }
    }

    // Could go back to using runtime selection of bindless or not, but using compiletime for now. Makes it less
    // portable but don't think it's that portable between systems anyways... (Without compiling on the system
    // ofcourse)

#else
    fprintf(stderr, "Running in CPU_ONLY mode\n");
    try
    {
        // For in order queue use this (usefull for debugging)
        // sycl::device cpu_dev = sycl::device{sycl::cpu_selector_v};
        // _device_queue = sycl::queue(
        //   cpu_dev, sycl::property_list{sycl::property::queue::in_order{},
        //   sycl::property::queue::enable_profiling{}});

        sycl::device dev = sycl::device{sycl::cpu_selector_v};
        _device_queue = sycl::queue(sycl::context{dev}, dev, queue_proplist);
    }
    catch(const sycl::exception& e)
    {
        std::cerr << "Failed to create CPU queue: " << e.what() << std::endl;
        throw;
    }

    // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
#endif
}

// PopSift::PopSift(const popsift::Config& config)
PopSift::PopSift(const popsift::Config& config, popsift::Config::ProcessingMode mode, ImageMode imode)
  : _image_mode(imode)
{
    initQueue();
    configure(config);

    if(imode == ByteImages) // default
    {
        // Push two images as we use two one to load in data and other to compute and they alter using the queue
        if constexpr(USE_BINDLESS_INPUT && sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                     sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d>())
        {
            // Swap out with ImageBindless when ready
            _pipe._unused.push(new popsift::ImageBindless(_device_queue));
            _pipe._unused.push(new popsift::ImageBindless(_device_queue));
        }
        else
        {
            _pipe._unused.push(new popsift::Image(_device_queue));
            _pipe._unused.push(new popsift::Image(_device_queue));
        }
    }
    else
    {
        // _pipe._unused.push(new popsift::ImageFloat);
        // _pipe._unused.push(new popsift::ImageFloat);
        // TODO: Add support for float images

        POP_FATAL("Currently not implemented");
    }

    _pipe._thread_stage1.reset(new std::thread(&PopSift::uploadImages, this));

    if(mode == popsift::Config::ExtractingMode)
        _pipe._thread_stage2.reset(new std::thread(&PopSift::extractDownloadLoop, this));
    else
        _pipe._thread_stage2.reset(new std::thread(&PopSift::matchPrepareLoop, this));

    sycl::device dev = _device_queue.get_device();
    if(dev.has(sycl::aspect::queue_profiling))
    {
        std::cout << "Queue profiling is supported.\n";
    }

    num_cu = dev.get_info<sycl::info::device::max_compute_units>(); // static public
    set_sg_per_cu();

// #if USE_JOINT_MATRIX && !CPU_ONLY // I
#if USE_JOINT_MATRIX // AMX (currently only on 4th, 5th and 6th generation xeon CPU's) does supoprt the matrix extension
    // Should be done before first call to match if not it will use normal version until it's true
    // This could be done at compile time if you pass the arcitecture from cmake check and use the compile time query
    // But I think in this case it does not matter much. As this way is more flexible (and easier to implement :D)
    // sycl::device dev = _device_queue.get_device();
    auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    auto max_cu = dev.get_info<sycl::info::device::max_compute_units>();
    auto max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    auto max_wi_dimensions = dev.get_info<sycl::info::device::max_work_item_dimensions>();

// There is an extension that is supposed to be more consistent and less ambigous, but I could not make it work
// auto num_cu_ext = dev.get_info<sycl::ext::oneapi::info::device::num_compute_units>();
// auto num_cu_ext = dev.get_info<sycl::info::device::num_compute_units>();
// auto num_cu_ext = sycl::ext::oneapi::info::device::num_compute_units()
#ifdef SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE
    printf("Device arhitectures are defined and can be used\n");

#endif

    printf("\n\nMax CU = %d -- Max WG = %zu -- WI dims = %d\n\n", max_cu, max_wg, max_wi_dimensions);
    for(const auto& combo : combinations)
    {
#if PRINT_MATRIX_OPTIONS
        std::cout << "M: " << combo.msize << ", N: " << combo.nsize << ", K: " << combo.ksize
                  << ", A type: " << matrix_type_to_string(combo.atype)
                  << ", B type: " << matrix_type_to_string(combo.btype)
                  << ", C type: " << matrix_type_to_string(combo.ctype)
                  << ", D type: " << matrix_type_to_string(combo.dtype) << "\n";
#endif

        if(combo.atype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp16 &&
           combo.btype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp16 &&
           combo.ctype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp32 &&
           combo.dtype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp32 && combo.msize == 16 &&
           combo.nsize == 16 && combo.ksize == 16)
        {
            matrixSupported = true;
            break; // No need to continue search
        }
    }
#endif
}

PopSift::~PopSift()
{
    if(_isInit)
    {
        uninit();
    }
}

bool PopSift::configure(const popsift::Config& config, bool /*force*/)
{
    if(_pipe._pyramid != nullptr)
    {
        return false;
    }

    _config = config;
    _config.levels = max(2, config.levels);

    return true;
}

void PopSift::uninit()
{
    if(!_isInit)
    {
        std::cerr << "[warning] Attempt to release resources from an uninitialized "
                     "instance"
                  << std::endl;
        return;
    }

    if(_d_gauss != nullptr)
        sycl::free(_d_gauss, _device_queue);
    else
        std::cout << "_d_gauss was a nullptr hennce not freeing" << std::endl;

    if(_d_consts != nullptr)
        sycl::free(_d_consts, _device_queue);
    else
        std::cout << "_d_consts was a nullptr hennce not freeing" << std::endl;

    _pipe.uninit();

    _isInit = false;
}

sycl::event PopSift::init_gauss_filter()
{
    // Crate gauss filter store it on host
    popsift::init_filter(_config, &_h_gauss);

    // Transfer gauss filter to device
    if(_d_gauss == nullptr)
    {
        _d_gauss = popsift::sycl_common::malloc_devT<popsift::GaussInfo>(
          1, __FILE__, __LINE__, "Failed to allocate gauss filter on device", _device_queue);
    }

#ifdef USE_PERSISTENT
    popsift::Pyramid::span = _h_gauss.dd.span[0];

    // const int span = d_gauss->dd.span[0];

#endif

    // Look into partial updates for this one (currently any change to config would be expensive...) but again who
    // updates config while it's running...
    return _device_queue.memcpy(_d_gauss, &_h_gauss, sizeof(popsift::GaussInfo));
}

sycl::event PopSift::init_constants()
{
    popsift::init_constants(_config.sigma,
                            _config.levels,
                            _config.getPeakThreshold(),
                            _config._edge_limit,
                            _config.getMaxExtrema(),
                            _config.getNormalizationMultiplier(),
                            &_h_consts);

    // Transfer constants to device
    if(_d_consts == nullptr)
    {
        _d_consts = popsift::sycl_common::malloc_devT<popsift::ConstInfo>(
          1, __FILE__, __LINE__, "Failed to allocate constants on device", _device_queue);
    }

    // Again look into partial update (but normaly updats won't be done)
    return _device_queue.memcpy(_d_consts, &_h_consts, sizeof(popsift::ConstInfo));
}

// Apply configuration should reside here
bool PopSift::applyConfiguration(bool force)
{
    if(force || (_config != _shadow_config))
    {
        // for re ren we need to free and re malloc or change the size or not malloc again if it is already malloced
        // _d_gauss_writPe = popsift::init_filter(_config, _config.sigma, _config.levels, _device_queue, &_d_gauss);
        _d_gauss_write = this->init_gauss_filter();

        _d_consts_write = this->init_constants();

        if(_config.getSgPerCu() != -1)
        {
            printf("Setting sub_group per Compute Unit value\n");
            this->sg_per_cu = _config.getSgPerCu();
        }
    }
    _shadow_config = _config;
    return true;
}

void PopSift::private_apply_scale_factor(int* w, int* h)
{
    /* up=-1 -> scale factor=2
     * up= 0 -> scale factor=1
     * up= 1 -> scale factor=0.5
     */
    float upscaleFactor = _config.getUpscaleFactor();
    float scaleFactor = 1.0f / powf(2.0f, -upscaleFactor);

    if(_config.octaves < 0)
    {
        int oct = max(int(floor(logf((float)min(*w, *h)) / logf(2.0f)) - 3.0f + scaleFactor), 1);
        _config.octaves = oct;
    }

    *w = ceilf(*w * scaleFactor);
    *h = ceilf(*h * scaleFactor);
}

void get_scale_factor(int* w, int* h, const float& upscaleFactor)
{
    float scaleFactor = 1.0f / powf(2.0f, -upscaleFactor);

    *w = ceilf(*w * scaleFactor);
    *h = ceilf(*h * scaleFactor);
}

bool PopSift::private_init(int w, int h)
{
    Pipe& p = _pipe;

    // WARNING: Already done to the job _w and _h in case of USM not Bindless
    // So incase of USM we are redoing the computation
    private_apply_scale_factor(&w, &h);

    if(p._pyramid != nullptr)
    {
        p._pyramid->resetDimensions(_config, w, h);
        return true;
    }

    p._pyramid = new popsift::Pyramid(_config, w, h, _device_queue, _d_gauss, _d_consts, _h_consts);

    return true;
}

// Don't see a purpose of returning true here as popsift did hence making it void
void PopSift::private_uninit()
{
    Pipe& p = _pipe;

    delete p._pyramid;
    p._pyramid = nullptr;
}

SiftJob* PopSift::enqueue(int w, int h, const unsigned char* imageData)
{
    // TODO(jorgejen): Implement support for float images and have this
    // configuration step if( _image_mode != ByteImages )
    // {
    //     stringstream ss;
    //     ss << "Image mode error" << endl
    //        << "E    Cannot load byte images into a PopSift pipeline
    //        configured for float images";
    //     POP_FATAL(ss.str());
    // }

    SiftJob* job = new SiftJob(w, h, imageData);
    _pipe._queue_stage1.push(job);
    return job;
}

void PopSift::uploadImages()
{
    SiftJob* job;
    while((job = _pipe._queue_stage1.pull()) != nullptr)
    {
        popsift::ImageBase* img = _pipe._unused.pull(); // getting a unused Image (reusing it)

        job->setImg(img, _config.getUpscaleFactor());

        _device_queue.wait();

        _pipe._queue_stage2.push(job);
    }
    // Push nullptr to stage2 queue to make that one terminates aswell
    // safe to do as we know know no more jobs will be pushed to stage 1 queue
    _pipe._queue_stage2.push(nullptr);
}

void PopSift::extractDownloadLoop()
{
    applyConfiguration(true); // Applies configuration is only run once as
                              // the thread is started
    Pipe& p = _pipe;

    SiftJob* job;
    while((job = p._queue_stage2.pull()) != nullptr)
    {
        // will do nothing if configuraiton has not changed
        applyConfiguration();

        popsift::ImageBase* img = job->getImg();

        private_init(img->getWidth(), img->getHeight());

        p._pyramid->step1(_config, img, _d_gauss_write, job->getImgTransferEvent());

        _device_queue.wait(); // SHould not be needed

#if QUEUE_PROFILING
        double frame_start =
          p._pyramid->_input_horiz_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
#endif

        // uploaded Image object is no longer needed, release for reuse
        p._unused.push(img);

        p._pyramid->step2(_config, _d_consts_write);

        // Copy featrues to host -- step 3
        popsift::FeaturesHost* features = p._pyramid->get_descriptors(_config);

        bool log_to_file = (_config.getLogMode() == popsift::Config::All);
        if(log_to_file)
        {
            // Log to file functions
            // Missing function
        }

        // Fufill the promise
        _device_queue.wait_and_throw(); // SHoud use event to wait for last part

#if QUEUE_PROFILING
        double frame_end =
          p._pyramid->_final_desc_event.template get_profiling_info<sycl::info::event_profiling::command_end>();

        printf("\n\nFrame time = %lf nanoseconds == %lf ms\n\n",
               frame_end - frame_start,
               (frame_end - frame_start) / 1000000);

        // TODO: Store to file in csv of json format, STORE multiple timings per frame time from start to Gaussian is
        // done, Then to DoG is done, Then to extrema is done and then orientation ans so on. (Due to not using in-order
        // queue I don't know which will finish last so would probably need to track multiple and use the largest value
        // as end time)
#endif

        job->setFeatures(features);
    }

    // _device_queue.wait(); // Having a wait here before I have all events configured properly
    private_uninit();
}

void PopSift::matchPrepareLoop()
{
    applyConfiguration(true);

    Pipe& p = _pipe;

    SiftJob* job;
    while((job = p._queue_stage2.pull()) != nullptr)
    {
        popsift::FeaturesDev* features;
        try
        {
            applyConfiguration();

            popsift::ImageBase* img = job->getImg();
            // Should add imagebase and ImageFloat to support float images
            // popsift::ImageBase* img = job->getImg();

            private_init(img->getWidth(), img->getHeight());

            p._pyramid->step1(_config, img, _d_gauss_write, job->getImgTransferEvent());

            // uploaded Image object is no longer needed, release for reuse
            p._unused.push(img);

            p._pyramid->step2(_config, _d_consts_write);

            // There are wait's in step2 descriptor hence this works TODO: Replace with events

            features = p._pyramid->clone_device_descriptors(_config);
            _device_queue.wait(); // Should be removed and only depend on dependencies events
        }
        catch(const std::exception& e)
        {
            job->setError(std::current_exception());
            job->setFeatures(nullptr);
            break;
        }

#if USE_JOINT_MATRIX
        // Here we compute the squared norm of the descriptors
        // Matching functions using tensor will wait for thie do be done by event
        // Could add function to wait for it do be done if user is using it's own matching function

        features->compute_squared_norms();
#endif
        // Now squared norm is scheduled to be computed and we can fuill promise and move on to next
        job->setFeatures(features);
    }

    private_uninit();
}

SiftJob::SiftJob(int w, int h, const unsigned char* imageData)
  : _w(w)
  , _h(h)
{
    _f = _p.get_future(); // tie the future to the promise so that it can be
                          // retrieved when it is eventually set

    // copy the data from caller
    _imageData = (unsigned char*)malloc(w * h);
    if(_imageData != nullptr)
    {
        memcpy(_imageData, imageData, w * h);
    }
    else
    {
        std::stringstream ss;
        ss << "Memory limitation" << endl << "E    Failed to allocate memory for SiftJob";
        POP_FATAL(ss.str());
    }
}

SiftJob::~SiftJob() { free(_imageData); }

// Do we need dynamic cast
popsift::FeaturesHost* SiftJob::getHost() { return dynamic_cast<popsift::FeaturesHost*>(_f.get()); }

popsift::FeaturesDev* SiftJob::getDev()
{
    popsift::FeaturesBase* features = _f.get();
    if(this->_err != nullptr)
    {
        std::rethrow_exception(this->_err);
    }
    return dynamic_cast<popsift::FeaturesDev*>(features);
}

void SiftJob::setError(std::exception_ptr ptr) { this->_err = ptr; }

void SiftJob::setImg(popsift::ImageBase* img, const float upscaleFactor)
{
    img->resetDimensions(_w, _h, upscaleFactor);

    _img_transfer_event = img->load(_imageData);

    _img = img;
}

// Not sure if this is a good way of doing it
// just doing it to have same methods as popsift code
popsift::ImageBase* SiftJob::getImg() { return _img; }

void SiftJob::setFeatures(popsift::FeaturesBase* f) { _p.set_value(f); }

void PopSift::Pipe::uninit()
{
    _queue_stage1.push(nullptr);
    if(_thread_stage2 != nullptr)
    {
        _thread_stage2->join();
        _thread_stage2.reset(nullptr);
    }
    if(_thread_stage1 != nullptr)
    {
        _thread_stage1->join();
        _thread_stage1.reset(nullptr);
    }

    while(!_unused.empty())
    {
        popsift::ImageBase* img = _unused.pull();
        delete img;
    }
}
