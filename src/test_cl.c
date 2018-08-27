#include "test_cl.h"
#include "utils.h"

#include <stdio.h>
#include <time.h>

#include <CL/cl.h>
#include "define_cl.h"

void test_cl()
{
    size_t global[3] = { 1,1,1 };
    size_t local[3] = { 1,1,1 };

    printf("\n\nStart Test_cl\n");

    double time = what_time_is_it_now();
    cl_kernel test_kernel = clGetkrnl_test_cl();
    cl_run_kernel3d(test_kernel, global, local, 3);

    printf("Test_cl Predicted in %f seconds.\n\n", what_time_is_it_now()-time);
}