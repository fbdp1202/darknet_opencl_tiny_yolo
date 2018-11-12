#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
#include "define_cl.h"

#ifdef OPENCL
#include <CL/cl.h>
#endif // OPENCL

typedef layer convolutional_layer;

#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif
#endif

#ifdef OPENCL
void forward_convolutional_layer_ocl(convolutional_layer layer, network net);

#ifdef HALF_MODE
void conv_ocl_half(convolutional_layer l, network net, cl_mem *mo_in, cl_mem *mo_out, int vec_size);
#endif // HALF_MODE

#ifdef SHORT_MODE
void conv_ocl_short(convolutional_layer l, network net, cl_mem *mo_in, cl_mem *mo_out, int vec_size, int index);
#endif // SHORT_MODE

#ifdef FIXED_MODE
void conv_ocl_fixed(convolutional_layer l, network net, cl_mem *mo_in, cl_mem *mo_out, int vec_size);
#endif // FIXED_MODE

void conv_ocl_float(convolutional_layer l, network net, cl_mem *mo_in, cl_mem *mo_out, int vec_size);

#endif // OPENCL

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void forward_convolutional_layer(const convolutional_layer layer, network net);
void update_convolutional_layer(convolutional_layer layer, update_args a);
image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_weights);
void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(convolutional_layer *l);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void backward_convolutional_layer(convolutional_layer layer, network net);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_weight(convolutional_layer layer, int i);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);

#endif // CONVOLUTIONAL_LAYER_H