/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_constants.hpp"
#include "sycl/queue.hpp"

#include <iostream>
#include <vector>

namespace popsift {

// Forward declaration
struct Descriptor; // float features[128];

/**
 * @brief This is a data structure that is returned to a calling program.
 * The xpos/ypos information in feature is scale-adapted.
 */
struct Feature
{
    int debug_octave;
    float xpos;
    float ypos;
    /// scale
    float sigma;
    /// number of this extremum's orientations
    /// remaining entries in desc are 0
    int num_ori;
    float orientation[ORIENTATION_MAX_COUNT];
    Descriptor* desc[ORIENTATION_MAX_COUNT];

    void print(std::ostream& ostr, bool write_as_uchar) const;
};

std::ostream& operator<<(std::ostream& ostr, const Feature& feature);

class FeaturesBase
{
    int _num_ext;
    int _num_ori;

  public:
    FeaturesBase();
    virtual ~FeaturesBase();

    inline int size() const { return _num_ext; }
    inline int getFeatureCount() const { return _num_ext; }
    inline int getDescriptorCount() const { return _num_ori; }

    inline void setFeatureCount(int num_ext) { _num_ext = num_ext; }
    inline void setDescriptorCount(int num_ori) { _num_ori = num_ori; }
};

/**
 * @brief This is a data structure that is returned to a calling program.
 * _ori is a transparent flat memory holding descriptors
 * that are referenced by the extrema.
 *
 * Note that the current data structures do not allow to match
 * Descriptors in the transparent array with their extrema except
 * for brute force.
 *
 */
class FeaturesHost : public FeaturesBase
{
    sycl::queue _device_queue;
    Feature* _ext;
    Descriptor* _ori;

  public:
    FeaturesHost() = delete; // no default constructor - need the queue
    FeaturesHost(sycl::queue Q);
    FeaturesHost(sycl::queue Q, int num_ext, int num_ori);
    ~FeaturesHost() override;

    typedef Feature* F_iterator;
    typedef const Feature* F_const_iterator;

    inline F_iterator begin() { return _ext; }
    inline F_const_iterator begin() const { return _ext; }
    inline F_iterator end() { return &_ext[size()]; }
    inline F_const_iterator end() const { return &_ext[size()]; }

    void reset(int num_ext, int num_ori);
    // void pin();
    // void unpin();

    inline Feature* getFeatures() { return _ext; }
    inline Descriptor* getDescriptors() { return _ori; }

    void print(std::ostream& ostr, bool write_as_uchar) const;

  protected:
    friend class Pyramid; // Pyramid will have access to everything not just what is defined as protected the protected
                          // does nothing for this one as far as  I understand
};

// using Features = FeaturesHost;

std::ostream& operator<<(std::ostream& ostr, const FeaturesHost& feature);

class FeaturesDev : public FeaturesBase
{
    sycl::queue _device_queue;
    Feature* _ext;    // array of extrema
    Descriptor* _ori; // array of desciptors
    int* _rev;        // the reverse map from descriptors to extrema

  public:
    FeaturesDev() = delete;
    FeaturesDev(sycl::queue Q);
    FeaturesDev(sycl::queue Q, int num_ext, int num_ori);
    ~FeaturesDev() override;

    void reset(int num_ext, int num_ori);

    /** This function performs one-directional brute force matching on
     *  the GPU between the Descriptors in this objects and the object
     *  other.
     *  The resulting matches are printed.
     */
    void match(FeaturesDev* other);

    /** This function performs one-directional brute force matching on
     *  the GPU between the Descriptors in this objects and the object
     *  other.
     *  The resulting matches are returned in an array of int3 that must
     *  be released with a call to cudaFree().
     *  The length of the array is this->getDescriptorCount().
     *  For each element at position i
     *    i is the index of a descriptor in this->getDescriptors()
     *    int3.x is the index of the best match in other->getDescriptors()
     *    int3.y is the index of the second best match in other->getDescriptors()
     *    int3.z indicates if the match is valid (non-zero) or not (zero)
     */
    // int3* matchAndReturn(FeaturesDev* other);
    // sycl::vec<int, 3>* matchAndReturn(FeaturesDev* other);

    // void destroyMatchMatrix(sycl::vec<int, 3>* matrix_ptr)
    // {
    //     if(!matrix_ptr)
    //         return;
    //
    //     try
    //     {
    //         auto alloc_kind = sycl::get_pointer_type(ptr, q.get_context());
    //
    //         switch(alloc_kind)
    //         {
    //             case sycl::usm::alloc::shared: sycl::free(ptr, q); break;
    //             case sycl::usm::alloc::host:
    //             case sycl::usm::alloc::device:
    //                 std::cerr << "This is a host/device poitner and not what this function is designed to do. \nThis
    //                 function is only to free the passed sycl::vec<int, 3>* matrix pointer returned "
    //                              "by matchAndReturn\n";
    //                 break;
    //             case sycl::usm::alloc::unknown:
    //                 // Pointer was either already freed or not allocated via SYCL
    //                 if(verbose)
    //                     std::cerr << "Warning: Pointer not currently allocated via SYCL (may be already freed)\n";
    //                 break;
    //         }
    //     }
    //     catch(const sycl::exception& e)
    //     {
    //         std::cerr << "SYCL exception during free: " << e.what() << "\n";
    //     }
    //     catch(...)
    //     {
    //         std::cerr << "Unknown exception during free\n";
    //     }
    // }

    // Need to be freed with correct context so this will never work
    std::tuple<sycl::vec<int, 3>*, std::function<void()>, std::function<void()>> matchAndReturn(FeaturesDev* other);
    std::tuple<sycl::vec<int, 3>*, std::function<void()>, std::function<void()>> matrixMatchAndReturn(
      FeaturesDev* other);

    /** This function takes as parameters that matches returned by
     *  matchAndReturn and releases that memory.
     */
    // void freeMatches(int3* match_matrix);

    inline Feature* getFeatures() { return _ext; }
    inline Descriptor* getDescriptors() { return _ori; }
    inline int* getReverseMap() { return _rev; }

    Descriptor* getDescriptor(int descIndex);
    const Descriptor* getDescriptor(int descIndex) const;
    Feature* getFeatureForDescriptor(int descIndex);
    const Feature* getFeatureForDescriptor(int descIndex) const;
};

} // namespace popsift
