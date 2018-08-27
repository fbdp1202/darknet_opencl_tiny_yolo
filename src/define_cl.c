#include "define_cl.h"

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_program program = NULL;
cl_command_queue command_queue = NULL;

cl_kernel krnl_pool = NULL;
cl_kernel krnl_conv3 = NULL;
cl_kernel krnl_pool2 = NULL;
cl_kernel krnl_conv3_vec = NULL;
cl_kernel krnl_conv3_vec4 = NULL;
cl_kernel krnl_conv3_vec8 = NULL;
cl_kernel krnl_conv3_vec16 = NULL;

cl_mem d_a, d_b, d_c;

cl_mem clGetMem_d_a() { return d_a; }
cl_mem clGetMem_d_b() { return d_b; }
cl_mem clGetMem_d_c() { return d_c; }

cl_mem* clGetpMem_d_a() { return &d_a; }
cl_mem* clGetpMem_d_b() { return &d_b; }
cl_mem* clGetpMem_d_c() { return &d_c; }

void clSetupMem() {
	d_a = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * 32 * 144, NULL);
	d_b = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * 144 * 169, NULL);
	d_c = clCreateMemobj(CL_MEM_WRITE_ONLY, sizeof(float) * 32 * 169, NULL);
}

void clReleaseMem() {
	clFreeMemobj(d_a);
	clFreeMemobj(d_b);
	clFreeMemobj(d_c);
}

cl_kernel clGetkrnl_pool() { return krnl_pool; }
cl_kernel clGetkrnl_conv3() { return krnl_conv3; }
cl_kernel clGetkrnl_pool2() { return krnl_pool2; }
cl_kernel clGetkrnl_conv3_vec() {return krnl_conv3_vec; }
cl_kernel clGetkrnl_conv3_vec4() {return krnl_conv3_vec4; }
cl_kernel clGetkrnl_conv3_vec8() {return krnl_conv3_vec8; }
cl_kernel clGetkrnl_conv3_vec16() {return krnl_conv3_vec16; }

void Deivce_info(cl_platform_id platform, cl_device_id device){
	cl_int err;
	cl_char string[10240] = {0};

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(string), &string, NULL);
    printf("Platform: %s\n", string);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
    printf("Vendor: %s\n", string);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
    printf("Version: %s\n", string);

    printf("\t-------------------------\n");

    // Get device name
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(string), &string, NULL);
    printf("\t\tName: %s\n", string);

    // Get device OpenCL version
    err = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(string), &string, NULL);
    printf("\t\tVersion: %s\n", string);

    // Get Max. Compute units
    cl_uint num;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, NULL);
    printf("\t\tMax. Compute Units: %d\n", num);

    // Get local memory size
    cl_ulong mem_size;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
    printf("\t\tLocal Memory Size: %llu KB\n", mem_size/1024);

    // Get global memory size
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
    printf("\t\tGlobal Memory Size: %llu MB\n", mem_size/(1024*1024));

    // Get maximum buffer alloc. size
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size, NULL);
    printf("\t\tMax Alloc Size: %llu MB\n", mem_size/(1024*1024));

    // Get work-group size information
    size_t size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
    printf("\t\tMax Work-group Total Size: %ld\n", size);

    // Find the maximum dimensions of the work-groups
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
    // Get the max. dimensions of the work-groups
    size_t dims[num];
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);
    printf("\t\tMax Work-group Dims: ( ");
    size_t k;
    for (k = 0; k < num; k++)
    {
        printf("%ld ", dims[k]);
    }
    printf(")\n");

    printf("\t-------------------------\n");

}

char * getKernelSource(const char *filename)
{
	FILE *file = fopen(filename, "r");
	if (!file)
	{
		fprintf(stderr, "Error: Could not open kernel source file\n");
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	int len = ftell(file) + 1;
	rewind(file);

	char *source = (char *)calloc(sizeof(char), len);
	if (!source)
	{
		fprintf(stderr, "Error: Could not allocate memory for source string\n");
		exit(EXIT_FAILURE);
	}
	fread(source, sizeof(char), len, file);
	fclose(file);
	return source;
}

char const* clGetErrorString(cl_int const err) {
	switch (err)
	{
		CL_ERR_TO_STR(CL_SUCCESS);
		CL_ERR_TO_STR(CL_DEVICE_NOT_FOUND);
		CL_ERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
		CL_ERR_TO_STR(CL_OUT_OF_RESOURCES);
		CL_ERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
		CL_ERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_MEM_COPY_OVERLAP);
		CL_ERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
		CL_ERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
		CL_ERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_MAP_FAILURE);
		CL_ERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
		CL_ERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
		CL_ERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
		CL_ERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_INVALID_VALUE);
		CL_ERR_TO_STR(CL_INVALID_DEVICE_TYPE);
		CL_ERR_TO_STR(CL_INVALID_PLATFORM);
		CL_ERR_TO_STR(CL_INVALID_DEVICE);
		CL_ERR_TO_STR(CL_INVALID_CONTEXT);
		CL_ERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
		CL_ERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
		CL_ERR_TO_STR(CL_INVALID_HOST_PTR);
		CL_ERR_TO_STR(CL_INVALID_MEM_OBJECT);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_SIZE);
		CL_ERR_TO_STR(CL_INVALID_SAMPLER);
		CL_ERR_TO_STR(CL_INVALID_BINARY);
		CL_ERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_PROGRAM);
		CL_ERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_NAME);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
		CL_ERR_TO_STR(CL_INVALID_KERNEL);
		CL_ERR_TO_STR(CL_INVALID_ARG_INDEX);
		CL_ERR_TO_STR(CL_INVALID_ARG_VALUE);
		CL_ERR_TO_STR(CL_INVALID_ARG_SIZE);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_ARGS);
		CL_ERR_TO_STR(CL_INVALID_WORK_DIMENSION);
		CL_ERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
		CL_ERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
		CL_ERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
		CL_ERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
		CL_ERR_TO_STR(CL_INVALID_EVENT);
		CL_ERR_TO_STR(CL_INVALID_OPERATION);
		CL_ERR_TO_STR(CL_INVALID_GL_OBJECT);
		CL_ERR_TO_STR(CL_INVALID_BUFFER_SIZE);
		CL_ERR_TO_STR(CL_INVALID_MIP_LEVEL);
		CL_ERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
		CL_ERR_TO_STR(CL_INVALID_PROPERTY);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
		CL_ERR_TO_STR(CL_INVALID_COMPILER_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);
		//CL_ERR_TO_STR(CL_INVALID_PIPE_SIZE);
		//CL_ERR_TO_STR(CL_INVALID_DEVICE_QUEUE);

	default:
		return "UNKNOWN ERROR CODE";
	}
}

void clSetup(const char *krnl_file) {
	cl_int err;
	char log[CL_LOG_SIZE] = { 0 };
	size_t log_size;

	//platform
	cl_uint ret_num_platforms;
	err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if (err != CL_SUCCESS) {
		printf("Error: no platforms available or OpenCL install broken");
		exit(EXIT_FAILURE);
	}

	//device
	cl_uint ret_num_devices;
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to get the number of devices: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	Deivce_info(platform_id, device_id);

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a compute context! : %s\n", clGetErrorString(err));
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}

	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a command queue! : %s\n", clGetErrorString(err));
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}

	printf("create command_queue\n");

	char *krnl_bin;
	krnl_bin = getKernelSource(krnl_file);

	program = clCreateProgramWithSource(context, 1, (const char **)&krnl_bin, NULL, &err);
	if ((!program) || (err != CL_SUCCESS)) {
		printf("Error: Failed to create compute program %s\n", clGetErrorString(err));
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}
	printf("create Program\n");

	err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
	//err = clBuildProgram(program, 1, &device_id, "-cl-nv-maxrregcount=1024", NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to build program executable! : %s\n", clGetErrorString(err));

		err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &log_size);
		if (err != CL_SUCCESS)
			printf("Error: Failed to load err log string! : %s\n", clGetErrorString(err));
		else
			printf("%s", log);

		exit(EXIT_FAILURE);
	}
	printf("clBuildProgram\n");

	krnl_pool = clCreateKernel(program, "Pool", &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create kernel for pool: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	krnl_conv3 = clCreateKernel(program, "Conv3", &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create kernel for conv3: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	krnl_conv3_vec = clCreateKernel(program, "Conv3_vec", &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create kernel for conv3_vec: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	krnl_conv3_vec4 = clCreateKernel(program, "Conv3_vec4", &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create kernel for conv3_vec4: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	krnl_conv3_vec8 = clCreateKernel(program, "Conv3_vec8", &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create kernel for conv3_vec8: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	krnl_conv3_vec16 = clCreateKernel(program, "Conv3_vec16", &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create kernel for conv3_vec16: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	krnl_pool2 = clCreateKernel(program, "Pool2", &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create kernel for pool2: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("INFO: complete to make CL properties\n");
}

void clReleaseAll() {
	clFlush(command_queue);
	clFinish(command_queue);
	
	clReleaseKernel(krnl_pool);
	clReleaseKernel(krnl_conv3);
	clReleaseKernel(krnl_pool2);
	clReleaseKernel(krnl_conv3_vec);
	clReleaseKernel(krnl_conv3_vec4);
	clReleaseKernel(krnl_conv3_vec8);

	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

void clSetKrnlArg(cl_kernel krnl, cl_uint num, size_t size, void *ptr) {
	int err = clSetKernelArg(krnl, num, size, ptr);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arg %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

cl_mem clCreateMemobj(cl_mem_flags flags, size_t size, float* host_ptr) {
	int errNum = 0;
	cl_mem mem = clCreateBuffer(context, flags, size, host_ptr, &errNum);

	if (!mem || errNum != CL_SUCCESS) {
		printf("Error: Failed to allocate device memory!: %s\n", clGetErrorString(errNum));
		exit(EXIT_FAILURE);
	}

	return mem;
}

void clFreeMemobj(cl_mem buffer) {
	clReleaseMemObject(buffer);
}

void cl_memcpy_to_device(cl_mem dest, void* src,
	size_t size) {
	int err = clEnqueueWriteBuffer(command_queue, dest, CL_TRUE, 0, size,
		src, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to write to source array a!\n");
		exit(EXIT_FAILURE);
	}
}

void cl_memcpy_from_device(void* dest, cl_mem src,
	size_t size) {
	int err = clEnqueueReadBuffer(command_queue, src, CL_TRUE, 0, size,
		dest, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array! %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void cl_run_kernel3d(cl_kernel krnl, size_t* global, size_t* local, cl_uint workDim) {
	cl_event event;

	int err = clEnqueueNDRangeKernel(command_queue, krnl, workDim,
		NULL, global, local, 0, NULL, &event);
	if (err != CL_SUCCESS) {
		printf("Error: failed to execute kernel! %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	clWaitForEvents(1, &event);
	clReleaseEvent(event);
}

void cl_run_kernel3d_async(cl_kernel krnl, size_t* global, size_t* local, cl_uint workDim) {

	int err = clEnqueueNDRangeKernel(command_queue, krnl, workDim,
		NULL, global, local, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: failed to execute kernel! %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#ifndef INFINITY
#define INFINITY 1.0/0.0
#endif

#ifndef NAN
#define NAN 0.0/0.0
#endif

typedef union 
{
  int32_t i;
  float f;
} FloatConvUnion;

cl_half poclu_float_to_cl_half(float value) 
{
  FloatConvUnion u;
  u.f = value;
  cl_half half = (u.i >> 16) & 0x8000; // sign
  cl_half fraction = (u.i >> 12) & 0x007ff; // fraction with extra bit for rounding
  cl_half exponent = (u.i >> 23)  & 0xff; // exponent
  
  if(exponent < 0x0067) // Return signed zero if zero or value is too small for denormal half
    return half;

  if(exponent > 0x008e){// value was NaN or Inf
    half |= 0x7c00u; // Make into inf
    half |= exponent == 255 && (u.i & 0x007fffffu); // If value was NaN make this into NaN
    return half;
  }

  if(exponent < 0x0071){// Denormal
    fraction |= 0x0800u;

    // rounding
    half |= (fraction >> (0x0072 - exponent)) + ((fraction >> (0x0071 - exponent)) & 1);
    return half;
  }

  half |= ((exponent - 0x0070) << 10) | (fraction >> 1);
  half += fraction & 1;// rounding
  return half;
}

float poclu_cl_half_to_float(cl_half value) 
{
  if (value == 0xFC00) {
    return -INFINITY;
  }
  if (value == 0x7C00) {
    return INFINITY;
  }

  int sgn = ((value & 0x8000) >> 15);
  int exp = (value & 0x7C00) >> 10;
  int mant = value & 0x03FF;

  if (exp == 0x1F && mant != 0) {
    return NAN;
  }

  float v = (exp == 0) ? mant : mant | 0x0400; // 1.x if not denormal
  v /= 0x400;
  float mul = exp2((float)exp - 15);
  v *= mul;
  if (sgn) {
    v *= -1;
  }
  return v;
}

void do_conversion_h_to_f(float *to, cl_half *from, int size){
    int i;
    for(i = 0; i<size; i++){
        to[i] = poclu_cl_half_to_float(from[i]);
    }
}

void do_conversion_f_to_h(cl_half *to, float *from, int size){
    int i;
    for(i=0; i<size; i++){
        to[i] = poclu_float_to_cl_half(from[i]);
    }
}

