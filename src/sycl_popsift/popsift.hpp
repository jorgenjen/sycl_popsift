#pragma once

#include "common/debug_macros.hpp"
#include "common/sync_queue.h"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/s_image.hpp"
#include "sycl_popsift/sift_constants.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

#include <sycl/sycl.hpp>

#include <future>
#include <sstream>
#include <thread>

namespace popsift {
class Image;
class Pyramid;

// template<class T>
// T* malloc_devT(int num, const char* file, int line, sycl::queue Q);

};

class SiftJob
{
    // TODO(jorgejen): Look into using these feature constructs
    // std::promise<popsift::FeaturesBase*> _p;
    // std::future <popsift::FeaturesBase*> _f;
    // NOTE: Using these simple integer features to have same pattern as popsift
    std::promise<int> _p;
    std::future<int> _f;

    int _w;
    int _h;

    unsigned char* _imageData; // Copies image to this from caller (I guess due to not
                               // trusting calleer and host performance is a non issue)

    popsift::Image* _img;
    std::exception_ptr _err;
    sycl::event _img_transfer_event; // pointer to event that we update

    friend class PopSift;

  public:
    /**
     * @brief Constructor for byte images, value range 0..255
     * @param[in] w the width in pixel of the image
     * @param[in] h the height in pixel of the image
     * @param[in] imageData the image buffer
     */
    SiftJob(int w, int h, const unsigned char* imageData);

    /**
     * @brief Destructor releases all the resources.
     */
    ~SiftJob();

    /**
     * @brief
     * @return
     */
    // popsift::FeaturesHost* getHost();
    // popsift::FeaturesDev*  getDev();

    int getHost(); // currently using int to have same pattern of
                   // initialization and such

    void setImg(popsift::Image* img, sycl::queue q, const float& upscaleFactor);

    popsift::Image* getImg();

    inline sycl::event getImgTransferEvent() { return _img_transfer_event; }

    // NOTE: Temporary to fufill the promise see popsift later on for proper
    // implmentation and do that
    void jobDone(int tmpRes);
    /** fulfill the promise */
    // void setFeatures( popsift::FeaturesBase* f );
    //
    // void setError(std::exception_ptr ptr);

    // TMP just for testing structure
    void printJob();
};

class PopSift
{
    struct Pipe
    {
        std::unique_ptr<std::thread> _thread_stage1;
        std::unique_ptr<std::thread> _thread_stage2;
        popsift::SyncQueue<SiftJob*> _queue_stage1;
        popsift::SyncQueue<SiftJob*> _queue_stage2;
        popsift::SyncQueue<popsift::Image*> _unused; // this is a comment

        popsift::Pyramid* _pyramid{nullptr};

        /**
         * @brief Release the allocated resources, if any.
         */
        void uninit();
    };

  public:
    // Constructors that are not allowed
    PopSift() = delete;
    PopSift(const PopSift&) = delete;

    // Constructors
    explicit PopSift(const popsift::Config& config);

    // should be one more for float images

    // destructor
    ~PopSift();

    /**
     * @brief Provide the configuration if you used the PopSift default
     *  constructor
     */
    bool configure(const popsift::Config& config, bool force = false);

    /**
     * @brief Release the resources.
     */
    void uninit();

    SiftJob* enqueue(int w, int h, const unsigned char* imageData);

  private:
    void printDim();
    void printDevice();
    void modifyImage();
    void printImageRegion(sycl::range<2> horiz, sycl::range<2> vert);

    bool applyConfiguration(bool force = false);

    bool private_init(int w, int h);
    void private_uninit();
    void private_apply_scale_factor(int* w, int* h);

    void extractDownloadLoop();
    void uploadImages();

    inline void initQueue();

    // destructor
    // ~PopSift();

    // Private attributes:
    int _w;
    int _h;
    // sycl::buffer<unsigned char, 2> _imageData;
    sycl::queue _device_queue;
    popsift::GaussInfo* _d_gauss = nullptr;
    popsift::ConstInfo* _d_consts = nullptr;

    sycl::event _d_gauss_write;
    sycl::event _d_consts_write;

    popsift::Config _config;

    /* Keep a copy of the config to avoid unnecessary re-configurations
     * in configure()
     */
    popsift::Config _shadow_config; // Not sure if I want this

    Pipe _pipe;
    /// whether the object is initialized
    bool _isInit{true};
};
