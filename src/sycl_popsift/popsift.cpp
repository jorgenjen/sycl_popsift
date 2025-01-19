#include "popsift.hpp"
#include "sycl/accessor.hpp"
#include "common/debug_macros.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"

#include <sycl/sycl.hpp>
#include <iostream>
#include <cstring>
#include <sstream>


using namespace std;

PopSift::PopSift(const popsift::Config& config)
{

  // should use the confige here to configure but requires that you have the pyramid and all that


  // sycl::queue _device_queue;
  // sycl::buffer<unsigned char, 2> _imageData(imageData, sycl::range<2>(_w, _h));

  cout << "PopSift constructor" << endl;

  // Push two images as we use two one to load in data and other to compute and they alter using the queue
  _pipe._unused.push( new popsift::Image(_device_queue));
  _pipe._unused.push( new popsift::Image(_device_queue));


  // TODO: Setup these threads
  _pipe._thread_stage1.reset( new std::thread( &PopSift::uploadImages, this ));
  _pipe._thread_stage2.reset( new std::thread( &PopSift::extractDownloadLoop, this ));

  // NOTE: Currently not supporting extraction and config like that
  // _pipe._thread_stage1.reset( new std::thread( &PopSift::uploadImages, this ));
  // if( mode == popsift::Config::ExtractingMode )
  //   _pipe._thread_stage2.reset( new std::thread( &PopSift::extractDownloadLoop, this ));
  // else
  //   _pipe._thread_stage2.reset( new std::thread( &PopSift::matchPrepareLoop, this ));


}
PopSift::~PopSift()
{
    if(_isInit)
    {
        uninit();
    }
}

void PopSift::uninit( )
{
    if(!_isInit)
    {
        std::cerr << "[warning] Attempt to release resources from an uninitialized instance" << std::endl;
        return;
    }
    std::cout << "Uninting the pipe now" << std::endl;
    _pipe.uninit();
    std::cout << "Done with the pip epipe now" << std::endl;

    _isInit = false;
}



void PopSift::printDim()
{
  cout << "Width: " << _w << endl;
}

void PopSift::printDevice()
{
  {
    using namespace sycl;

    try {
      // queue q& = _device_queue;


      std::cout << "Selected device in PopSift method using SYCL: "
        << _device_queue.get_device().get_info<info::device::name>()
        << "\n";
    } catch (const sycl::exception& e) {
      std::cout << "Exception caught: " << e.what() << std::endl;
    }

  }
}


// Helper function for development
// ranges are inclusive on 0th dimension and exclusive on 1th dimension
// void PopSift::printImageRegion(sycl::range<2> horiz, sycl::range<2> vert)
// {
//   // print out the first 10 bytes of the image
//   using namespace sycl;
//
//   // wait for all previous enqued task to end before doing the print to show desired data
//   _device_queue.wait();
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
//   std::cout << "Image region: horiz = (" << horiz.get(0) << " -> " << horiz.get(1)
//             << ") vert = (" << vert.get(0) << " -> " << vert.get(1) << ")" << std::endl;
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
//     std::cout << "Selected device in PopSift method (modifyImage) using SYCL: "
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

SiftJob* PopSift::enqueue( int                  w,
                           int                  h,
                           const unsigned char* imageData )
{
    // TODO: Implement support for float images and have this configuration step
    // if( _image_mode != ByteImages )
    // {
    //     stringstream ss;
    //     ss << "Image mode error" << endl
    //        << "E    Cannot load byte images into a PopSift pipeline configured for float images";
    //     POP_FATAL(ss.str());
    // }

    // HERE TEXTURE FIT WAS DONE -- NOt currently using texture memory

    SiftJob* job = new SiftJob( w, h, imageData );
    // NOTE: Currently skipping first stage as I don't have explicit memory tranfer from host to device first
    _pipe._queue_stage1.push( job );
    return job;
}

void PopSift::uploadImages( )
{
  SiftJob* job;
  while( ( job = _pipe._queue_stage1.pull() ) != nullptr ) {
    popsift::Image* img = _pipe._unused.pull(); // getting a unused Image (reusing it)

    // copy image to device
    job->setImg( img, _device_queue );
    // WARNING: the copy is asynchronous so could result in issues as the job could start before it is done 
    // but I think as the following tasks are also in the same sycl queue it should be fine due to it making the task 
    // graph properly and that the memcoyp is a dependency but not sure might have to set it to be a dependency before the 
    // next dependent kernel runs...


    // job->setImg( img );
    _pipe._queue_stage2.push( job );
  }
  // Push nullptr to stage2 queue to make that one terminates aswell
  // safe to do as we know know no more jobs will be pushed to stage 1 queue
  _pipe._queue_stage2.push( nullptr );
}


void PopSift::extractDownloadLoop( )
{
    // cudaSetDevice(_device);
    // applyConfiguration(true); // Applies configuration is only run once as the thread is started

    std::cout << "Starting download loop thread" << std::endl;
    Pipe& p = _pipe;

    SiftJob* job;
    while( ( job = p._queue_stage2.pull() ) != nullptr ) {
        // get the next job in queue or wait until a new job arrives

    popsift::Image* img = job->getImg();
    std::cout << "the job is --> ";
    job->printJob();


    // DO THE JOB!!!


    // FUFULL THE PROMISE

    job->jobDone(5);


    p._unused.push( img ); // uploaded input image no longer needed, release for reuse




    // applyConfiguration();

    // popsift::ImageBase* img = job->getImg();

    // TODO: Do similar private init and appyl scalefactor that this method calls
    // private_init( img->getWidth(), img->getHeight() );

  }
}



SiftJob::SiftJob( int w, int h, const unsigned char* imageData )
  : _w(w)
  , _h(h)
// , _img(nullptr) // Currently not in use
{
  _f = _p.get_future(); // tie the future to the promise so that it can be retrieved when it is eventually set

  // copy the data from caller 
  _imageData = (unsigned char*)malloc( w*h ); 
  if( _imageData != nullptr )
  {
    memcpy( _imageData, imageData, w*h );
  }
  else {
    stringstream ss;
    ss << "Memory limitation" << endl
      << "E    Failed to allocate memory for SiftJob";
    POP_FATAL(ss.str());
  }
}

SiftJob::~SiftJob( )
{
  free( _imageData );
}

// To fufill promise temporary promise solution while I don't have a featuresBase object to return
void SiftJob::jobDone( int tmpRes )
{
    _p.set_value( tmpRes );
}


// TMP function for testing structure
void SiftJob::printJob()
{
  using namespace std;

  printf("Width: %d -- height: %d\n", _w, _h);

}

int SiftJob::getHost()
{
  return _f.get();
}

void SiftJob::setImg( popsift::Image* img, sycl::queue q )
{
    img->resetDimensions( _w, _h );
    img->load( _imageData );
    _img = img;

}

// Not sure if this is a good way of doing it
// just doing it to have same methods as popsift code
popsift::Image* SiftJob::getImg()
{
  return _img;
}


void PopSift::Pipe::uninit()
{
  _queue_stage1.push( nullptr );
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

  while( !_unused.empty() )
  {
    popsift::Image* img = _unused.pull();
    delete img;
  }
}

