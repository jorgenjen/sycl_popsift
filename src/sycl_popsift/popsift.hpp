#pragma once

#include <thread>
#include <sycl/sycl.hpp>

#include "common/sync_queue.h"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "s_image.hpp"


#include <future>


class SiftJob
{
  // TODO: Look into using these feature constructs
  // std::promise<popsift::FeaturesBase*> _p;
  // std::future <popsift::FeaturesBase*> _f;
  // NOTE: Using these simple integer features to have same pattern as popsift
  std::promise<int> _p;
  std::future<int> _f;

  int                 _w;
  int                 _h;
  unsigned char*      _imageData; // Copies image to this from caller (I guess due to not trusting calleer and host performance is a non issue)

  popsift::Image* _img; 
  std::exception_ptr _err;

public:
  /**
   * @brief Constructor for byte images, value range 0..255
   * @param[in] w the width in pixel of the image
   * @param[in] h the height in pixel of the image
   * @param[in] imageData the image buffer
   */
  SiftJob( int w, int h, const unsigned char* imageData );

  /**
   * @brief Destructor releases all the resources.
   */
  ~SiftJob( );

  /**
   * @brief
   * @return
   */
  // popsift::FeaturesHost* getHost();
  // popsift::FeaturesDev*  getDev();

  int getHost(); // currently using int to have same pattern of initialization and such


  void setImg(popsift::Image* img, sycl::queue q);

  popsift::Image* getImg();


  // NOTE: Temporary to fufill the promise see popsift later on for proper implmentation and do that
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
    std::unique_ptr<std::thread>            _thread_stage1;
    std::unique_ptr<std::thread>            _thread_stage2;
    popsift::SyncQueue<SiftJob*>            _queue_stage1;
    popsift::SyncQueue<SiftJob*>            _queue_stage2;
    popsift::SyncQueue<popsift::Image*> _unused;

    // popsift::Pyramid*                      _pyramid{nullptr};

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
   * @brief Release the resources.
   */
  void uninit( );

  /**
   * @brief Provide the configuration if you used the PopSift default
   *  constructor
   */
  bool configure( const popsift::Config& config, bool force = false );



  void printDim();
  void printDevice();
  void modifyImage();
  void printImageRegion(sycl::range<2> horiz, sycl::range<2> vert);



  SiftJob* enqueue(int w, int h, const unsigned char* imageData);
  void extractDownloadLoop();
  void uploadImages();
  // destructor
  // ~PopSift();

private:
  int _w;
  int _h;
  // sycl::buffer<unsigned char, 2> _imageData;
  sycl::queue _device_queue;



  Pipe _pipe; 
  /// whether the object is initialized
  bool            _isInit{true};


};


