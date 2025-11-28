## Where we tested the code

1. The branch `feature/evaluate-sift-descriptors` was used to run the sift versions and evaluate them with distance metrics
2. The branch `feature/multi-wg-matrix-matchng` was used to test both the matching performance

The same test code was used on PopSift with minor modifications to make it work with CUDA and the slight differences in API for sycl_popsift and PopSift

The main head does not have working sift which was broken due to the persistent threads experient that we merged in. Hence testing of sift from states before that. 

`Bin_counter` branch is the one used to retrive number of non -INFINITY in the sorting stage. 

There are some options in the cmake.
```bash
cmake ..  -DCMAKE_CXX_COMPILER=icpx -DENABLE_CUDA=ON -DJointMatrix=off -DProfiling=OFF -DPopSift_USE_GRID_FILTER=OFF -DOptimistic_ori_scheduling=off -DPersistentThreads=OFF -DPerfTestingFunctions=OFF -DPopSift_EXPERIMENTAL=ONA
```
These are all the options They do not all do anything however. The important ones are `JointMatrix` to toggle that on and of for matching. And `ENABLE_CUDA` to toggle GPU/CPU mode 
there is also `PopSift_EXPERIMENTAL` that toggles experiemntal features off like jointmatrix and bindless iamges



