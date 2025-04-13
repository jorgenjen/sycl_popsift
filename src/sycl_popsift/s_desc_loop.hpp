// // #include "sycl_popsift/sift_octave.hpp"
// #include "sycl_popsift/sift_pyramid.hpp"
//
// #undef BLOCK_3_DIMS
//
// namespace popsift {
//
// // Forward declarations
// // class ExtremaCounters;
// // class ExtremaBuffers;
// // class DevBuffers;
// // struct Extremum;
// // struct Descriptor;
//
// // class Ext_desc_loop
// // {
// //   private:
// //     ExtremaCounters* dct;
// //     ExtremaBuffers* dbuf;
// //     DevBuffers* dobuf;
// //     float** data;
// //     const int octave;
// //     const int width;
// //     const int height;
// //
// //   public:
// //     Ext_desc_loop(ExtremaCounters* dct,
// //                   ExtremaBuffers* dbuf,
// //                   DevBuffers* dobuf,
// //                   float** data,
// //                   const int octave,
// //                   const int width,
// //                   const int height);
// //
// //     inline void operator()(sycl::nd_item<3> it) const;
// // };
//
// class Octave; // forward declaration
//
// // Moved to Pyramid class so that I have access to hct without passing as param
// inline void Pyramid::start_ext_desc_loop(const int octave, Octave& oct_obj)
// {
//     // dim3 block;
//     // dim3 grid;
//     // grid.x = _hct.ori_ct[octave]; // orientation count for the octave
//     // grid.y = 1;
//     // grid.z = 1;
//
//     if(_hct.ori_ct[octave] == 0)
//         return;
//
// #ifndef BLOCK_3_DIMS
//     // block.x = 32;
//     // block.y = 4;
//     // block.z = 4;
//
//     sycl::range global{4, 4, static_cast<size_t>(_hct.ori_ct[octave])};
//     sycl::range local{4, 4, 32};
// #else
//     // block.x = 32;
//     // block.y = 1;
//     // block.z = 16;
//
//     sycl::range global{16, 1, _hct.ori_ct[octave]};
//     sycl::range local{16, 1, 32};
// #endif
//
//     // ext_desc_loop<<<grid, block, 0, oct_obj.getStream()>>>(
//     //   octave, oct_obj.getDataTexPoint(), oct_obj.getWidth(), oct_obj.getHeight());
//
//     _device_queue.parallel_for(
//       sycl::nd_range{global, local},
//       Ext_desc_loop(_dct, _dbuf, _dobuf, oct_obj.getDataArray(), octave, oct_obj.getWidth(), oct_obj.getHeight()));
//
//     // octave, oct_obj.getDataArray(), oct_obj.getWidth(), oct_obj.getHeight()));
//
//     // POP_SYNC_CHK;
//
//     // return true;
// }
// }; // namespace popsift
