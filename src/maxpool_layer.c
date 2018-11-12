#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>
#include <CL/cl.h>
#include "define_cl.h"
image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + 2*padding)/stride;
    l.out_h = (h + 2*padding)/stride;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + 2*l->pad)/l->stride;
    l->out_h = (h + 2*l->pad)/l->stride;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

void print_in_out(float *in, float *out, int w){
	printf("%f %f %f %f\n%f %f %f %f\n---------------\n", in[0], in[1], in[2], in[3], in[w], in[w+1], in[w+2], in[w+3]);
	printf("------out-------\n%f %f\n%f %f\n", out[0],out[1], out[w/2], out[w/2+1]);
	return;
}


void maxpool_ocl(const maxpool_layer l, network net, cl_mem *mo_in, cl_mem *mo_out)
{
//    printf("<forward_maxpooling_layer_opencl>\n");

	size_t global[3] = { l.w, l.h, l.c };
    //printf(" max pooling.. %.3f %.3f %.3f %.3f\n", net.input[0], net.input[1], net.input[2], net.input[3]);
    //cl_memcpy_to_device(*mo_out, net.input, sizeof(float) * l.w*l.h*l.c);
	if(net.index < 10){
		size_t local[3] = {26, 2, 1};
		clSetKernelArg(clGetkrnl_pool(), 0, sizeof(cl_mem), mo_in);
		clSetKernelArg(clGetkrnl_pool(), 1, sizeof(cl_mem), mo_out);
		cl_run_kernel3d(clGetkrnl_pool(), global, local, 3);
	}
	else{
		size_t local[3] = {13, 13, 1};
		clSetKernelArg(clGetkrnl_pool2(), 0, sizeof(cl_mem), mo_in);
		clSetKernelArg(clGetkrnl_pool2(), 1, sizeof(cl_mem), mo_out);
		cl_run_kernel3d(clGetkrnl_pool2(), global, local, 3);
	}
}


void testing(float *a, float *b, int size){
	printf("testing.........\n");
    int i;
	for(i=0; i<size; i++) if(a[i] != b[i]){ printf("no!!! cpu [%d] : %f, gpu [%d] : %f\n", i, a[i], i, b[i]); if(i>15) break;}
}

void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad;
    int h_offset = -l.pad;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}


void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

