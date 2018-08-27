#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define VERSION_GAMMA
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define CL_ERR_TO_STR(err) case err: return #err
#define CL_LOG_SIZE 8196
char const* clGetErrorString(cl_int const err);

cl_kernel clGetkrnl_conv();
cl_kernel clGetkrnl_conv2();
cl_kernel clGetkrnl_conv3();
cl_kernel clGetkrnl_conv3_vec();
cl_kernel clGetkrnl_conv3_vec4();
cl_kernel clGetkrnl_conv3_vec8();
cl_kernel clGetkrnl_conv3_vec16();
cl_kernel clGetkrnl_pool();
cl_kernel clGetkrnl_pool2();

cl_mem clGetMem_d_a();
cl_mem clGetMem_d_b();
cl_mem clGetMem_d_c();
cl_mem* clGetpMem_d_a();
cl_mem* clGetpMem_d_b();
cl_mem* clGetpMem_d_c();

void clSetup(const char *krnl_file);
void clReleaseAll();
void clSetupMem();
void clReleaseMem();

void clSetKrnlArg(cl_kernel krnl, cl_uint num, size_t size, void *ptr);

cl_mem clCreateMemobj(cl_mem_flags flags, size_t size, float* host_ptr);
void clFreeMemobj(cl_mem buffer);

void cl_memcpy_to_device(cl_mem dest, void* src, size_t size);
void cl_memcpy_from_device(void* dest, cl_mem src, size_t size);
void cl_run_kernel3d(cl_kernel krnl, size_t* global, size_t* local, cl_uint workDim);
void cl_run_kernel3d_async(cl_kernel krnl, size_t* global, size_t* local, cl_uint workDim);

float poclu_cl_half_to_float(cl_half value);
cl_half poclu_float_to_cl_half(float value); 
