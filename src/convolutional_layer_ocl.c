#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#include <CL/cl.h>
#include "define_cl.h"

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifdef _OCL

void forward_convolutional_layer_ocl(convolutional_layer l, network net, cl_mem *mo_in, cl_mem *mo_out, int depth)
{
    int LOCAL_DEPTH = depth;

    int m = l.n / l.groups;
    int k = l.size*l.size*l.c / l.groups;
    cl_kernel krnl_to_execute = clGetkrnl_conv3();
    printf("l.out_w:%d, l.out_h:%d\n",l.out_w, l.out_h);
    size_t global[3] = { l.out_w, l.out_h, (int)(l.out_c/LOCAL_DEPTH) };
    size_t local[3] = { 13, 13, 1 };
    
    double time = what_time_is_it_now();

    if(net.index == 13) clFreeMemobj(*mo_out);
    
    *mo_out = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(float) * l.out_w * l.out_h * l.out_c, NULL);
    cl_mem mo_filt = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * m*k, NULL);
    cl_mem mo_mean = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * m, NULL);
    cl_mem mo_variance = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * m, NULL);
    cl_mem mo_scales = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * m, NULL);
    cl_mem mo_biases = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * m, NULL);

    cl_memcpy_to_device(mo_filt, l.weights, sizeof(float) * m*k);
    cl_memcpy_to_device(mo_biases, l.biases, sizeof(float) * m);
    cl_memcpy_to_device(mo_mean, l.rolling_mean, sizeof(float) * m);
    cl_memcpy_to_device(mo_variance, l.rolling_variance, sizeof(float) * m);
    cl_memcpy_to_device(mo_scales, l.scales, sizeof(float) * m);

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), mo_in);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &mo_filt);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), mo_out);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &l.w);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &l.h);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &l.c);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &l.out_c);
    clSetKernelArg(krnl_to_execute, 7, sizeof(cl_mem), &mo_mean);
    clSetKernelArg(krnl_to_execute, 8, sizeof(cl_mem), &mo_variance);
    clSetKernelArg(krnl_to_execute, 9, sizeof(cl_mem), &mo_scales);
    clSetKernelArg(krnl_to_execute, 10, sizeof(cl_mem), &mo_biases);
    printf("alloc done\n");
    cl_run_kernel3d(krnl_to_execute, global, local, 3);

    if(net.index == 12){   // move C to A
        global[2] = l.out_c;
        clSetKernelArg(clGetkrnl_move(), 0, sizeof(cl_mem), mo_in);
        clSetKernelArg(clGetkrnl_move(), 1, sizeof(cl_mem), mo_out);
        clSetKernelArg(clGetkrnl_move(), 2, sizeof(int), &l.out_c);
        cl_run_kernel3d(clGetkrnl_move(), global, local, 3);
    }
    // todo : swap mo_in and mo_out without the move kernel.
}

#endif
