#pragma once

#include <thread>

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
  explicit PopSift(int w, int h);

  void printDim();
  void printDevice();

  // destructor
  // ~PopSift();

private:
  int _w;
  int _h;

};

