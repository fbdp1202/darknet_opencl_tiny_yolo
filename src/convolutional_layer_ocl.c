#ifdef OPENCL
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
#endif // AI2

#ifdef HALF_MODE
void conv_ocl_half(convolutional_layer l, network net, cl_mem *mo_in, cl_mem *mo_out, int vec_size)
{   
//    printf("<forward_convolutional_layer_opencl>\n");

//    int LOCAL_DEPTH = 1;
    int m = l.n / l.groups;
    int k = l.size*l.size*l.c / l.groups;
    cl_kernel krnl_to_execute = clGetkrnl_conv3();
    
    if(net.index > 11){
        if      (vec_size == 1) krnl_to_execute = clGetkrnl_conv3();
        else if (vec_size == 2) krnl_to_execute = clGetkrnl_conv3_vec();
        else if (vec_size == 4) krnl_to_execute = clGetkrnl_conv3_vec4();
        else if (vec_size == 8) krnl_to_execute = clGetkrnl_conv3_vec8();
        //krnl_to_execute = clGetkrnl_conv3_vec();
    }

    /////////////
    size_t global[3] = { l.out_w, l.out_h, (int)(l.out_c/LOCAL_DEPTH) };
    size_t local[3] = { 13, 13, 1 };

    cl_mem mo_mean, mo_variance, mo_scales, mo_filt, mo_biases;

    mo_filt     = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(cl_half) * l.nweights, NULL);
    mo_biases   = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(cl_half) * l.n, NULL);
    cl_memcpy_to_device(mo_filt,     l.weights_half,          sizeof(cl_half) * l.nweights );
    cl_memcpy_to_device(mo_biases,   l.biases_half,           sizeof(cl_half) * l.n );


    if (l.batch_normalize && (!l.dontloadscales)){
        mo_mean     = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(cl_half) * l.n, NULL);
        mo_variance = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(cl_half) * l.n, NULL);
        mo_scales   = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(cl_half) * l.n, NULL);
        cl_memcpy_to_device(mo_mean,     l.rolling_mean_half,     sizeof(cl_half) * l.n );
        cl_memcpy_to_device(mo_variance, l.rolling_variance_half, sizeof(cl_half) * l.n );
        cl_memcpy_to_device(mo_scales,   l.scales_half,           sizeof(cl_half) * l.n );
    }
    
    double time = what_time_is_it_now();

    clSetKernelArg(krnl_to_execute, 0,  sizeof(cl_mem),     mo_in);
    clSetKernelArg(krnl_to_execute, 1,  sizeof(cl_mem),     &mo_filt);
    clSetKernelArg(krnl_to_execute, 2,  sizeof(cl_mem),     mo_out);
    clSetKernelArg(krnl_to_execute, 3,  sizeof(int),        &l.w);
    clSetKernelArg(krnl_to_execute, 4,  sizeof(int),        &l.h);
    clSetKernelArg(krnl_to_execute, 5,  sizeof(int),        &l.c);
    clSetKernelArg(krnl_to_execute, 6,  sizeof(int),        &l.out_c);
    clSetKernelArg(krnl_to_execute, 7,  sizeof(cl_mem),     &mo_mean);
    clSetKernelArg(krnl_to_execute, 8,  sizeof(cl_mem),     &mo_variance);
    clSetKernelArg(krnl_to_execute, 9,  sizeof(cl_mem),     &mo_scales);
    clSetKernelArg(krnl_to_execute, 10, sizeof(cl_mem),     &mo_biases);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

    clFreeMemobj(mo_mean);
    clFreeMemobj(mo_variance);
    clFreeMemobj(mo_scales);
    clFreeMemobj(mo_filt);
    clFreeMemobj(mo_biases);
}
#endif

#ifdef FIXED_MODE
void conv_ocl_fixed(convolutional_layer l, network net, cl_mem *mo_in, cl_mem *mo_out, int idx)
{
    int m = l.n / l.groups;
    int k = l.size*l.size*l.c / l.groups;
    int n = l.out_w*l.out_h;
    int on14x14 = 1;
    cl_kernel krnl_to_execute;
    if(on14x14) krnl_to_execute = clGetkrnl_conv3_14x14();
    else        krnl_to_execute = clGetkrnl_conv3();

    size_t global[3] = { l.out_w, l.out_h, (int)(l.out_c/LOCAL_DEPTH) };
    size_t local[3] = { 13, 13, 1 };

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), mo_in);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), clGet_mo_filt_Mem(idx));
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), mo_out);    
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &l.w);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &l.h);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &l.c);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &l.out_c);
    clSetKernelArg(krnl_to_execute, 7, sizeof(cl_mem),  clGet_mo_mean_Mem(idx));
    clSetKernelArg(krnl_to_execute, 8, sizeof(cl_mem),  clGet_mo_variance_Mem(idx));
    clSetKernelArg(krnl_to_execute, 9, sizeof(cl_mem),  clGet_mo_scales_Mem(idx));
    clSetKernelArg(krnl_to_execute, 10, sizeof(cl_mem), clGet_mo_biases_Mem(idx));
    clSetKernelArg(krnl_to_execute, 11, sizeof(float), &l.fixed_config[0]);
    clSetKernelArg(krnl_to_execute, 12, sizeof(float), &l.fixed_config[1]);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);
}

#endif // FIXED_MODE

void conv_ocl_float(convolutional_layer l, network net, cl_mem *mo_in, cl_mem *mo_out, int vec_size)
{
    int m = l.n / l.groups;
    int k = l.size*l.size*l.c / l.groups;
    cl_kernel krnl_to_execute = clGetkrnl_conv3();

    size_t global[3] = { l.out_w, l.out_h, (int)(l.out_c/LOCAL_DEPTH) };
    size_t local[3] = { 13, 13, 1 };

    
    if(vec_size == -2){
        krnl_to_execute = clGetkrnl_in_conv3();
    }
    if(vec_size == -3){
        krnl_to_execute = clGetkrnl_in_ConvMax();
        local[0] = 16;
        local[1] = 16;
    }
    if(vec_size == -4){
        krnl_to_execute = clGetkrnl_Second_ConvMax();
        local[0] = 8;
        local[1] = 8;
    }

    if(l.w == 13)
        krnl_to_execute = clGetkrnl_in_conv3_13x13();

    if(net.index > 11){
        if(vec_size == 2) krnl_to_execute = clGetkrnl_conv3_vec();
        else if(vec_size == 4) krnl_to_execute = clGetkrnl_conv3_vec4();
        else if(vec_size == 8) krnl_to_execute = clGetkrnl_conv3_vec8();
        //krnl_to_execute = clGetkrnl_conv3_vec();
    }

    /////////////
    cl_mem mo_filt = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * m * k, NULL);
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

    cl_run_kernel3d(krnl_to_execute, global, local, 3);
}

#endif // OPENCL