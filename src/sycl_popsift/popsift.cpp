#include "popsift.hpp"
#include "sycl/accessor.hpp"
#include "common/debug_macros.hpp"

#include <sycl/sycl.hpp>
#include <iostream>
#include <cstring>
#include <sstream>


using namespace std;

PopSift::PopSift(int w, int h, unsigned char* imageData)
  : _w(w),
  _h(h),
  // _imageData(imageData, sycl::range<2>(w, h), {sycl::property::buffer::use_host_ptr()})
  _imageData(imageData, sycl::range<2>(w, h))
{

  // sycl::queue _deviceQueue;
  // sycl::buffer<unsigned char, 2> _imageData(imageData, sycl::range<2>(_w, _h));

  cout << "PopSift constructor" << endl;

    // CUrrently only supporting ByteImages and not using _unused as I don't know it's purpose yet...
//   if( imode == ByteImages )
//   {
//     _pipe._unused.push( new popsift::Image);
//     _pipe._unused.push( new popsift::Image);
//   }
//   else
// {
//     _pipe._unused.push( new popsift::ImageFloat );
//     _pipe._unused.push( new popsift::ImageFloat );
//   }

  // TODO: Setup these threads
  // _pipe._thread_stage1.reset( new std::thread( &PopSift::uploadImages, this ));
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
    _pipe.uninit();

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
      // queue q& = _deviceQueue;


      std::cout << "Selected device in PopSift method using SYCL: "
        << _deviceQueue.get_device().get_info<info::device::name>()
        << "\n";
    } catch (const sycl::exception& e) {
      std::cout << "Exception caught: " << e.what() << std::endl;
    }

  }
}


// Helper function for development
// ranges are inclusive on 0th dimension and exclusive on 1th dimension
void PopSift::printImageRegion(sycl::range<2> horiz, sycl::range<2> vert)
{
  // print out the first 10 bytes of the image
  using namespace sycl;

  // wait for all previous enqued task to end before doing the print to show desired data
  _deviceQueue.wait();

  host_accessor<unsigned char, 2, access::mode::read> h_acc(_imageData);

  if (vert.get(0) > _w && vert.get(0) < 0 || 
      vert.get(1) > _w && vert.get(1) < 0 ||
      vert.get(0) >= vert.get(1)
  )
  {
    std::cout << "Image region is not legal" << std::endl;
  }

  std::cout << "Image region: horiz = (" << horiz.get(0) << " -> " << horiz.get(1)
            << ") vert = (" << vert.get(0) << " -> " << vert.get(1) << ")" << std::endl;
  // using range in a odd way (I know :D)
  for (int i = vert.get(0); i < vert.get(1); ++i)
  {
    for (int j = horiz.get(0); j < horiz.get(1); ++j)
    {
         std::printf("%03u ", h_acc[j][i]);
    }
       std::cout << std::endl;
  }
}


void PopSift::modifyImage()
{
  using namespace sycl;
  try {

    std::cout << "Selected device in PopSift method (modifyImage) using SYCL: "
      << _deviceQueue.get_device().get_info<info::device::name>()
      << "\n";
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Modify the image
  std::cout << "Modifyig image now" << std::endl;

  _deviceQueue.submit([&](handler& cgh) {
    printf("w=%d  -- h=%d", _w, _h);

    accessor img(_imageData, cgh, read_write);
    cgh.parallel_for(range<2>(_w, _h), [=](id<2> idx) {
      img[idx] = img[idx] - 1;
    });
  });

}

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
    _pipe._queue_stage2.push( job );
    return job;
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

        std::cout << "the job is --> ";
        job->printJob();
        

        // DO THE JOB!!!




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








// should be called as part of cleanup of popsift
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
        _thread_stage1->join();
        _thread_stage1.reset(nullptr);
    }

    // while( !_unused.empty() )
    // {
    //     popsift::ImageBase* img = _unused.pull();
    //     delete img;
    // }
}

