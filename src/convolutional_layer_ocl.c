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




#endif // OPENCL