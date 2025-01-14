#pragma once

#include <thread>
#include <sycl/sycl.hpp>

class PopSift
{
public:

  // Attributes
  struct Pipe
  {
    std::unique_ptr<std::thread>            _thread_stage1;
    std::unique_ptr<std::thread>            _thread_stage2;
  };
  

  // Constructors
  explicit PopSift(int w, int h, unsigned char* imageData);

  void printDim();
  void printDevice();
  void modifyImage();
  void printImage();
  // destructor
  // ~PopSift();

private:
  int _w;
  int _h;
  sycl::buffer<unsigned char, 2> _imageData;
  sycl::queue _deviceQueue;

};

