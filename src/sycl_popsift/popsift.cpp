#include "sycl_popsift/popsift.hpp"

#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/malloc_devt.hpp"
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
    try
    {
        // If there is no GPU it will throw exception and use CPU in catch
        sycl::device dev = sycl::device{sycl::gpu_selector_v};
        _device_queue = sycl::queue(sycl::context{dev}, dev);
        // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
    }
    catch(sycl::exception const& ex)
    {
        cout << "No GPU found falling back to CPU... Exception thrown: " << ex.what() << endl;

        sycl::device dev = sycl::device{sycl::cpu_selector_v};
        _device_queue = sycl::queue(sycl::context{dev}, dev);
        // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
    }
#else
    fprintf(stderr, "Running in CPU_ONLY mode\n");
    sycl::device dev = sycl::device{sycl::cpu_selector_v};
    _device_queue = sycl::queue(sycl::context{dev}, dev);
    // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
#endif
}

PopSift::PopSift(const popsift::Config& config)
{
    // should use the confige here to configure but requires that you have the
    // pyramid and all that

    initQueue();
    configure(config);

    // Set the static memer pointers to nullptr
    // _d_gauss = nullptr;

    // sycl::queue _device_queue;
    // sycl::buffer<unsigned char, 2> _imageData(imageData, sycl::range<2>(_w,
    // _h));

    // cout << "PopSift constructor" << endl;

    // Push two images as we use two one to load in data and other to compute
    // and they alter using the queue
    _pipe._unused.push(new popsift::Image(_device_queue));
    _pipe._unused.push(new popsift::Image(_device_queue));

    // TODO(jorgejen): Setup these threads.
    _pipe._thread_stage1.reset(new std::thread(&PopSift::uploadImages, this));
    _pipe._thread_stage2.reset(new std::thread(&PopSift::extractDownloadLoop, this));

    std::cout << "Running on: " << _device_queue.get_device().get_info<sycl::info::device::name>() << endl;

    // NOTE: Currently not supporting extraction and config like that
    // _pipe._thread_stage1.reset( new std::thread( &PopSift::uploadImages, this
    // )); if( mode == popsift::Config::ExtractingMode )
    //   _pipe._thread_stage2.reset( new std::thread(
    //   &PopSift::extractDownloadLoop, this ));
    // else
    //   _pipe._thread_stage2.reset( new std::thread(
    //   &PopSift::matchPrepareLoop, this ));
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

    // std::cout << "Before config" << std::endl;
    _config = config;
    // std::cout << "AFTER config" << std::endl;
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
    fprintf(stderr, "In init_gauss_filter()\n");

    // Crate gauss filter store it on host
    popsift::init_filter(_config, &_h_gauss);

    fprintf(stderr, "AT bottom off init_gauss_filter()\n");
    // Transfer gauss filter to device
    if(_d_gauss == nullptr)
    {
        _d_gauss = popsift::common_sycl::malloc_devT<popsift::GaussInfo>(
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
    fprintf(stderr, "in init contsatns\n");

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
        _d_consts = popsift::common_sycl::malloc_devT<popsift::ConstInfo>(
          1, __FILE__, __LINE__, "Failed to allocate constants on device", _device_queue);
    }
    else
        std::cout << "\n\n\t\t_d_consts is set -- no malloc needed\n\n" << std::endl; // reomve in future

    // Again look into partial update (but normaly updats won't be done)
    return _device_queue.memcpy(_d_consts, &_h_consts, sizeof(popsift::ConstInfo));
}

// Apply configuration should reside here
bool PopSift::applyConfiguration(bool force)
{
    // TODO: Figure out why on second image it returns mismatch on octaves when they in fact are the same
    // so something seems to be wrong here. Once figured out revert the equal function back to the commented out one
    if(force || (_config != _shadow_config))
    {
        // for re ren we need to free and re malloc or change the size or not malloc again if it is already malloced
        // _d_gauss_writPe = popsift::init_filter(_config, _config.sigma, _config.levels, _device_queue, &_d_gauss);
        _d_gauss_write = this->init_gauss_filter();

        _d_consts_write = this->init_constants();

        // _d_consts_write = popsift::init_constants(_config.sigma,
        //                                           _config.levels,
        //                                           _config.getPeakThreshold(),
        //                                           _config._edge_limit,
        //                                           _config.getMaxExtrema(),
        //                                           _config.getNormalizationMultiplier(),
        //                                           _device_queue,
        //                                           &_d_consts);
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

    cout << "The scale factor: " << scaleFactor << endl;

    if(_config.octaves < 0)
    {
        int oct = max(int(floor(logf((float)min(*w, *h)) / logf(2.0f)) - 3.0f + scaleFactor), 1);
        cout << "Octaves: " << oct << endl;
        _config.octaves = oct;
    }

    *w = ceilf(*w * scaleFactor);
    *h = ceilf(*h * scaleFactor);
}

// not connected to the class just namespace!
void get_scale_factor(int* w, int* h, const float& upscaleFactor)
{
    // float upscaleFactor = _config.getUpscaleFactor();
    float scaleFactor = 1.0f / powf(2.0f, -upscaleFactor);

    *w = ceilf(*w * scaleFactor);
    *h = ceilf(*h * scaleFactor);
}

bool PopSift::private_init(int w, int h)
{
    Pipe& p = _pipe;

    // cout << "\n\n\t\tPopSift::private_init(" << w << "," << h << ")" << endl;

    // WARNING: Already done to the job _w and _h if reverted uncomment!
    private_apply_scale_factor(&w, &h);

    // cout << "\n\n\t\tPopSift::after_scale_factor(" << w << "," << h << ")" << endl;

    // TODO(jorgejen): Implement pyramid

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
        job->setImg(img, _device_queue, _config.getUpscaleFactor());

        // job->setImg( img );
        _pipe._queue_stage2.push(job);
    }
    // Push nullptr to stage2 queue to make that one terminates aswell
    // safe to do as we know know no more jobs will be pushed to stage 1 queue
    _pipe._queue_stage2.push(nullptr);
}

void PopSift::extractDownloadLoop()
{
    // cudaSetDevice(_device);
    // std::cout << "Befoe apply conf conf dong" << std::endl;
    applyConfiguration(true); // Applies configuration is only run once as
    // the thread is started

    // std::cout << "Starting download loop thread" << std::endl;
    Pipe& p = _pipe;

    SiftJob* job;
    while((job = p._queue_stage2.pull()) != nullptr)
    {
        // will do nothing if configuraiton has not changed
        applyConfiguration();

        popsift::Image* img = job->getImg();

        job->printJob(); // Can probs remove

        private_init(img->getWidth(), img->getHeight());

        std::vector<sycl::event> dependencies =
          p._pyramid->step1(_config, img, _d_gauss_write, job->getImgTransferEvent());

        // FUFULL THE PROMISE

        cout << "Jobby: -- " << endl;
        job->printJob();

        // uploaded input image no longer needed, release for reuse
        p._unused.push(img);

        p._pyramid->step2(_config, dependencies, _d_consts_write);

        _device_queue.wait();
        fflush(stdout);
        fflush(stderr);
        job->jobDone(5);
    }

    // _device_queue.wait(); // Having a wait here before I have all events configured properly
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

// To fufill promise temporary promise solution while I don't have a
// featuresBase object to return
void SiftJob::jobDone(int tmpRes) { _p.set_value(tmpRes); }

// TMP function for testing structure
void SiftJob::printJob() { std::printf("Width: %d -- height: %d\n", _w, _h); }

int SiftJob::getHost() { return _f.get(); }

void SiftJob::setImg(popsift::Image* img, sycl::queue q, const float& upscaleFactor)
{
    int scaled_w = _w;
    int scaled_h = _h;
    get_scale_factor(&scaled_w, &scaled_h, upscaleFactor);
    img->resetDimensions(_w, _h, scaled_w, scaled_h);

    sycl::event src_img_transfer = img->copy_src_dev(_imageData);

    // img->load(_imageData);
    // img->load_divide(_imageData);
    // img->load_divide_point(_imageData, scaled_w);
    // _img_transfer_event = img->load_divide_linear(_imageData, scaled_w);
    _img_transfer_event = img->load_linear(scaled_w, src_img_transfer);
    _img = img;
}

// Not sure if this is a good way of doing it
// just doing it to have same methods as popsift code
popsift::Image* SiftJob::getImg() { return _img; }

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
        // should not really ever run as for stage2 to be nullptr
        // stage1 has to have already become nullptr hence not needed
        // as far as I understand...
        _thread_stage1->join();
        _thread_stage1.reset(nullptr);
    }

    while(!_unused.empty())
    {
        popsift::Image* img = _unused.pull();
        delete img;
    }
}

// Helper function for development
// ranges are inclusive on 0th dimension and exclusive on 1th dimension
// void PopSift::printImageRegion(sycl::range<2> horiz, sycl::range<2> vert)
// {
//   // print out the first 10 bytes of the image
//   using namespace sycl;
//
//   // wait for all previous enqued task to end before doing the print to show
//   desired data _device_queue.wait();
//
//   host_accessor<unsigned char, 2, access::mode::read> h_acc(_imageData);
//
//   if (vert.get(0) > _w && vert.get(0) < 0 ||
//       vert.get(1) > _w && vert.get(1) < 0 ||
//       vert.get(0) >= vert.get(1)
//   )
//   {
//     std::cout << "Image region is not legal" << std::endl;
//   }
//
//   std::cout << "Image region: horiz = (" << horiz.get(0) << " -> " <<
//   horiz.get(1)
//             << ") vert = (" << vert.get(0) << " -> " << vert.get(1) << ")" <<
//             std::endl;
//   // using range in a odd way (I know :D)
//   for (int i = vert.get(0); i < vert.get(1); ++i)
//   {
//     for (int j = horiz.get(0); j < horiz.get(1); ++j)
//     {
//          std::printf("%03u ", h_acc[j][i]);
//     }
//        std::cout << std::endl;
//   }
// }

// void PopSift::modifyImage()
// {
//   using namespace sycl;
//   try {
//
//     std::cout << "Selected device in PopSift method (modifyImage) using SYCL:
//     "
//       << _device_queue.get_device().get_info<info::device::name>()
//       << "\n";
//   } catch (const sycl::exception& e) {
//     std::cout << "Exception caught: " << e.what() << std::endl;
//   }
//
//   // Modify the image
//   std::cout << "Modifyig image now" << std::endl;
//
//   _device_queue.submit([&](handler& cgh) {
//     printf("w=%d  -- h=%d", _w, _h);
//
//     accessor img(_imageData, cgh, read_write);
//     cgh.parallel_for(range<2>(_w, _h), [=](id<2> idx) {
//       img[idx] = img[idx] - 1;
//     });
//   });
//
// }
