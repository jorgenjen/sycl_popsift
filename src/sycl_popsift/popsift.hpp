#pragma once

#include "common/debug_macros.hpp"
#include "common/sync_queue.h"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/s_image.hpp"
#include "sycl_popsift/sift_constants.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

#if PERF_TESTING_FUNCTIONS
#include <sycl_popsift/sift_desc_config.hpp> // For FeatureType
#endif

#include <sycl/sycl.hpp>

#include <future>
#include <sstream>
#include <thread>

namespace popsift {
class ImageBase;
class Pyramid;

// template<class T>
// T* malloc_devT(int num, const char* file, int line, sycl::queue Q);

};

class SiftJob
{
    std::promise<popsift::FeaturesBase*> _p;
    std::future<popsift::FeaturesBase*> _f;

    int _w;
    int _h;

    unsigned char* _imageData; // Copies image to this from caller (I guess due to not
                               // trusting calleer and host performance is a non issue)

    popsift::ImageBase* _img;
    std::exception_ptr _err;
    sycl::event _img_transfer_event; // pointer to event that we update

    friend class PopSift; // Gives popsift class full access to SiftJob class

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
    popsift::FeaturesHost* getHost();
    popsift::FeaturesDev* getDev();

    void setImg(popsift::ImageBase* img, const float upscaleFactor);

    popsift::ImageBase* getImg();

    inline sycl::event getImgTransferEvent() { return _img_transfer_event; }

    /** fulfill the promise */
    void setFeatures(popsift::FeaturesBase* f);

    void setError(std::exception_ptr ptr);
};

class PopSift
{
    struct Pipe
    {
        std::unique_ptr<std::thread> _thread_stage1;
        std::unique_ptr<std::thread> _thread_stage2;
        popsift::SyncQueue<SiftJob*> _queue_stage1;
        popsift::SyncQueue<SiftJob*> _queue_stage2;
        popsift::SyncQueue<popsift::ImageBase*> _unused;

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

    /**
     * @brief Image modes
     */
    enum ImageMode
    {
        ///  byte image, value range 0..255
        ByteImages,
        /// float images, value range [0..1[
        FloatImages // Currently not supported
    };

    /**
     * @brief
     * @param config
     * @param mode
     * @param imode
     */
    explicit PopSift(const popsift::Config& config,
                     popsift::Config::ProcessingMode mode = popsift::Config::ExtractingMode,
                     ImageMode imode = ByteImages);

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

    // matrix is wheter or not to use matrix version if compiling with JointMatrix=ON when running cmake then
    // matrix=false will run normal mode but with sycl::half/fp16 to get float results configure with JointMatrix off
    // and set matrix to false
    // ouput_file is a csv formatted file
    void benchmarkMatchingPerformance(bool matrix,
                                      int seed,
                                      const std::string& img_dir_l,
                                      const std::string& img_dir_r,
                                      const std::string& output_filename);

    inline static bool matrixSupported = false;
    inline static int sg_per_cu = -1; // sub-group per execution-unit -- Start as not defined(-1)
    inline static int num_cu;

#if PERF_TESTING_FUNCTIONS
    void benchmarkMatchingPerformance(bool matrix, int seed, std::vector<std::array<FeatureType, 128>> desc_pool);
#endif

#if USE_PERSISTENT

    inline int max_span()
    {
        // _h_gauss.inc.span[_h_gauss.required_filter_stages + 2] should always be largest I think
        // The final if always largest aswell as it's based on the sigma which grows with the level

        // return max(_h_gauss.dd.span[0], _h_gauss.inc.span[_h_gauss.required_filter_stages + 2]);

        for(int i = 0; i <= _config.levels + 2; ++i)
        {
            std::printf("lvl = %d -- span = %d\n", i, _h_gauss.inc.span[i]);
        }

        return _h_gauss.inc.span[_config.levels + 2];
    }
#endif

  private:
    void modifyImage();
    void printImageRegion(sycl::range<2> horiz, sycl::range<2> vert);

    bool applyConfiguration(bool force = false);

    bool private_init(int w, int h);
    void private_uninit();
    void private_apply_scale_factor(int* w, int* h);

    void extractDownloadLoop();
    void uploadImages();
    void matchPrepareLoop();

    inline void initQueue();
    sycl::event init_gauss_filter();
    sycl::event init_constants();

    // destructor
    // ~PopSift();

    // Private attributes:
    int _w;
    int _h;
    sycl::queue _device_queue;

    popsift::GaussInfo* _d_gauss = nullptr;
    popsift::GaussInfo _h_gauss{};
    sycl::event _d_gauss_write;

    popsift::ConstInfo* _d_consts = nullptr;
    popsift::ConstInfo _h_consts{};
    sycl::event _d_consts_write;

    popsift::Config _config;

    /* Keep a copy of the config to avoid unnecessary re-configurations
     * in configure()
     */
    popsift::Config _shadow_config;

    ImageMode _image_mode;

    Pipe _pipe;

    void set_sg_per_cu();

    /// whether the object is initialized
    bool _isInit{true};
};
