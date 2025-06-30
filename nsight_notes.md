## Nsight Compute (ncu)

To run nsight compute profiling with some extra sections (not sure how to get cycles as that is gone with this one)
```bash
sudo LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/linux/lib:$LD_LIBRARY_PATH /usr/local/cuda-12.1/bin/ncu --metrics sm__cycles_elapsed.avg,sm__cycles_elapsed.sum,sm__inst_executed.sum --section
 Occupancy --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis --section LaunchStats --target-processes all -o profile_vert_wave_small ./popsift-demo --input-file /home/jorgenjensvold/Downloads/h
annover_2048_1365_img_only/img1.ppm
```

Then open file in nsight compute with: 
```bash
ncu-ui profile_vert_wave_small.ncu_rep
```


## Nsight systems

```bash
nsys profile .popsift-demo --input-file /path/to/imgs/
```


