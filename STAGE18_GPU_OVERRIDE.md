# Stage 18 GPU Resource Override

Stage 18A ran on physical GPU 3 as originally registered.

Before Stage 18B, two full 60-second GPU 3 resource windows observed an unknown
compute process and the task stopped without launching a competing job. The
user then explicitly authorized physical GPU 2 for the remaining Stage 18
experiments.

From Stage 18B onward:

- physical GPU: 2;
- `CUDA_VISIBLE_DEVICES=2`;
- internal program GPU: 0;
- data-loader workers: 1;
- the same 60-second free-memory, utilization, unknown-process, and
  single-MOSI-process gates continue to apply to GPU 2;
- no process on GPU 3 is modified or terminated.
