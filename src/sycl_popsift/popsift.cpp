#include "sycl_popsift/popsift.hpp"

#include "sycl/device.hpp"
#include "sycl/device_selector.hpp"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/features.hpp"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/sift_constants.hpp"

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

inline void PopSift::initQueue()
{
#ifndef CPU_ONLY
    // should probably also have a compile time flag --experimental to enable this feature
    if constexpr(sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_image_array>())
    {
        // Running with bindless image -- need to find gpu with that aspect (needed incase of multi gpu system)

        for(sycl::device dev : sycl::device::get_devices(sycl::info::device_type::gpu))
        {
            // Find GPU with the aspect (incase of multigpu system)
            if(dev.has(sycl::aspect::ext_oneapi_bindless_images) && dev.has(sycl::aspect::ext_oneapi_image_array))
            {
                std::cout << "Running on: " << _device_queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl
                          << "\t--> supports ext_oneapi_bindless_images: YES" << std::endl
                          << "\t--> supports ext_oneapi_image_array: YES" << std::endl
                          << std::endl;

                _device_queue = sycl::queue(sycl::context{dev}, dev);
                return; // We always select first gpu that had the aspect (might be a way to select the best one)
                        // but most systems will be single gpu anyways
            }
        }
        // Did not return hence we did not find a matching device to that was on the compiled system
        POP_FATAL("Could not find device with support for  ext_oneapi_bindless_images and ext_oneapi_image_array "
                  "Such a device was available at compile time... Please re-compile")
    }
    else if constexpr(sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>())
    {
        // In case it only supports bindless we can use it for upscaling still
        for(sycl::device dev : sycl::device::get_devices(sycl::info::device_type::gpu))
        {
            // Find GPU with the aspect (incase of multigpu system)
            if(dev.has(sycl::aspect::ext_oneapi_bindless_images))
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

                _device_queue = sycl::queue(sycl::context{dev}, dev);
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
            _device_queue = sycl::queue(sycl::context{dev}, dev);
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
    // sycl::device dev = sycl::device{sycl::cpu_selector_v};
    // _device_queue = sycl::queue(sycl::context{dev}, dev);

    // _device_queue =
    //   sycl::queue{sycl::cpu_selector_v, sycl::property::queue::in_order{},
    //   sycl::property::queue::enable_profiling{}};

    // _device_queue =
    //   sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order{},
    //   sycl::property::queue::enable_profiling{});

    try
    {
        // sycl::device cpu_dev = sycl::device{sycl::cpu_selector_v};
        // _device_queue = sycl::queue(
        //   cpu_dev, sycl::property_list{sycl::property::queue::in_order{},
        //   sycl::property::queue::enable_profiling{}});

        sycl::device dev = sycl::device{sycl::cpu_selector_v};
        _device_queue = sycl::queue(sycl::context{dev}, dev);
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
        _pipe._unused.push(new popsift::Image(_device_queue));
        _pipe._unused.push(new popsift::Image(_device_queue));
    }
    else
    {
        // _pipe._unused.push(new popsift::ImageFloat);
        // _pipe._unused.push(new popsift::ImageFloat);
        // TODO Add support fro float images
        fprintf(stderr, "Currently not implemented\n");
    }

    _pipe._thread_stage1.reset(new std::thread(&PopSift::uploadImages, this));

    if(mode == popsift::Config::ExtractingMode)
        _pipe._thread_stage2.reset(new std::thread(&PopSift::extractDownloadLoop, this));
    else
        _pipe._thread_stage2.reset(new std::thread(&PopSift::matchPrepareLoop, this));
}

PopSift::~PopSift()
{
    fprintf(stderr, "\n\tDESTROYING POPSIFT CLASS\n");
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

    // Uncommeetn for now not in use an global...
    // if(popsift::d_consts != nullptr)
    //     sycl::free(popsift::d_consts, _device_queue);
    // else
    //     std::cout << "d_consts was a nullptr hennce not freeing" << std::endl;

    fprintf(stderr, "\n\tINSIDE UNINIT OF POPSIFT\n");

    if(_d_gauss != nullptr)
        sycl::free(_d_gauss, _device_queue);
    else
        std::cout << "_d_gauss was a nullptr hennce not freeing" << std::endl;

    fprintf(stderr, "\n\tFreed _d_gauss -- next is _d_consts = %p\n", _d_consts);

    popsift::ConstInfo* me_consts = _d_consts;
    // _device_queue
    //   .single_task([=]() {
    //       sycl::ext::oneapi::experimental::printf("\n\n\tinside uninit _d_donsts norm_multi %d -- edge_limit
    //       %f\n",
    //                                               me_consts->norm_multi,
    //                                               me_consts->edge_limit);
    //   })
    //   .wait();

    if(_d_consts != nullptr)
        sycl::free(_d_consts, _device_queue);
    else
        std::cout << "_d_consts was a nullptr hennce not freeing" << std::endl;

    fprintf(stderr, "\n\tFreed _d_consts\n");

    _pipe.uninit();

    fprintf(stderr, "\n\tUninted the pipe\n");

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
    else
        std::cout << "\n\n\t\td_gauss is set -- no malloc needed\n\n" << std::endl; // Remove down the line

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
        cout << "\tNot null ptr" << endl;
        p._pyramid->resetDimensions(_config, w, h);
        return true;
    }

    p._pyramid = new popsift::Pyramid(_config, w, h, _device_queue, _d_gauss, _d_consts, _h_consts);

    return true;
}

// Don't see a purpose of returning true here as popsift did hence making it void
void PopSift::private_uninit()
{
    fprintf(stderr, "\npriv unint\n");
    Pipe& p = _pipe;

    delete p._pyramid;
    p._pyramid = nullptr;
}

void PopSift::printDim() { cout << "Width: " << _w << endl; }

void PopSift::printDevice()
{
    {
        try
        {
            // queue q& = _device_queue;

            std::cout << "Selected device in PopSift method using SYCL: "
                      << _device_queue.get_device().get_info<sycl::info::device::name>() << "\n";
        }
        catch(const sycl::exception& e)
        {
            std::cout << "Exception caught: " << e.what() << std::endl;
        }
    }
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

    // HERE TEXTURE FIT WAS DONE -- NOt currently using texture memory

    SiftJob* job = new SiftJob(w, h, imageData);
    _pipe._queue_stage1.push(job);
    return job;
}

void PopSift::uploadImages()
{
    SiftJob* job;
    while((job = _pipe._queue_stage1.pull()) != nullptr)
    {
        popsift::Image* img = _pipe._unused.pull(); // getting a unused Image (reusing it)

        // WARNING: CHANGING WIDTH AND HEIGHT IN JOB BASED ON PRIVATE APPLY
        // COULD BE PROBLEMATIC DOWN THE LINE -- BE AWARE YOU ARE HERBY WARNED!
        // USING firend class so breaking encapsulateion... (should change this)
        // private_apply_scale_factor(&job->_w, &job->_h);

        // cout << "Updated w=" << job->_w << " and h=" << job->_h << endl;
        // copy image to device
        job->setImg(img, _config.getUpscaleFactor());
        fprintf(stderr, "After setImg");

        // job->setImg( img );
        _pipe._queue_stage2.push(job);
        // break;
    }
    fprintf(stderr, "\n\n\t\tDone uploading\n\n");
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

        popsift::Image* img = job->getImg();

        private_init(img->getWidth(), img->getHeight());

        p._pyramid->step1(_config, img, _d_gauss_write, job->getImgTransferEvent());

        _device_queue.wait(); // SHould not be needed

        // uploaded Image object is no longer needed, release for reuse
        p._unused.push(img);

        p._pyramid->step2(_config, _d_consts_write);

        // Copy featrues to host -- step 3
        popsift::FeaturesHost* features = p._pyramid->get_descriptors(_config);

        // popsift::FeaturesDev* for_funsies_ja =
        //   p._pyramid->clone_device_descriptors(_config); // Delete this line move to match loop

        bool log_to_file = (_config.getLogMode() == popsift::Config::All);
        if(log_to_file)
        {
            // Log to file functions
        }

        // Fufill the promise
        job->setFeatures(features);

        _device_queue.wait_and_throw();
        fprintf(stderr, "\n\tEverytying done now we shut down the shop\n");
        fflush(stdout);
        fflush(stderr);
        // job->jobDone(5);
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

            popsift::Image* img = job->getImg();
            // Should add imagebase and ImageFloat to support float images
            // popsift::ImageBase* img = job->getImg();

            private_init(img->getWidth(), img->getHeight());

            p._pyramid->step1(_config, img, _d_gauss_write, job->getImgTransferEvent());

            // uploaded Image object is no longer needed, release for reuse
            p._unused.push(img);

            p._pyramid->step2(_config, _d_consts_write);

            features = p._pyramid->clone_device_descriptors(_config);
            _device_queue.wait(); // Should be removed and only depend on dependencies events
        }
        catch(const std::exception& e)
        {
            job->setError(std::current_exception());
            job->setFeatures(nullptr);
            break;
        }

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

SiftJob::~SiftJob()
{
    fprintf(stderr, "\n\tDESTROYING SIFTJOB\n");
    free(_imageData);
}

// To fufill promise temporary promise solution while I don't have a
// featuresBase object to return
// void SiftJob::jobDone(int tmpRes) { _p.set_value(tmpRes); }

// TMP function for testing structure
void SiftJob::printJob() { std::printf("Width: %d -- height: %d\n", _w, _h); }

// int SiftJob::getHost() { return _f.get(); }
// int SiftJob::getHost() { return _f.get(); }

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

void SiftJob::setImg(popsift::Image* img, const float upscaleFactor)
{
    // Moved to alloc called in resetDimensions
    // int scaled_w = _w;
    // int scaled_h = _h;
    // get_scale_factor(&scaled_w, &scaled_h, upscaleFactor);

    img->resetDimensions(_w, _h, upscaleFactor);

    sycl::event src_img_transfer = img->copy_src_dev(_imageData);

    // img->load(_imageData);
    // img->load_divide(_imageData);
    // img->load_divide_point(_imageData, scaled_w);
    // _img_transfer_event = img->load_divide_linear(_imageData, scaled_w);

    // _img_transfer_event = img->load_linear(scaled_w, src_img_transfer);
    _img_transfer_event = img->load_linear(src_img_transfer);

    // _img_transfer_event.wait();
    // fprintf(stderr, "\n\tWe got past sending og image to device!!! and doing load lienar\n");
    _img = img; // Why are you copying the image class pointer?
}

// Not sure if this is a good way of doing it
// just doing it to have same methods as popsift code
popsift::Image* SiftJob::getImg() { return _img; }

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
        popsift::Image* img = _unused.pull();
        delete img;
    }
}

void PopSift::allMainThread()
{
    // Seems to be fine  with thread setup :D

    // requires break in uploadImages to exit
    uploadImages();

    extractDownloadLoop();
    _device_queue.wait_and_throw();
}
