#ifdef OPENCL 

#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"
#include <CL/cl.h>
#include "define_cl.h"

#include "test_cl.h"

int find_out_size(layer l)
{
    return l.out_c * l.out_w * l.out_h;
}

int find_input_size(layer l)
{
    return l.c * l.w * l.h;
}

int find_max_out_size(network *netp)
{
    int i = 0, sz_out = 0;
    network net = *netp;

    for(i = 0; i < net.n; i++){
        int n2 = find_out_size(net.layers[i]);
        if( n2 > sz_out) sz_out = n2;
    }
    return sz_out;
}

void weight_reorder(convolutional_layer l){
    int i, j, k, w;
    int m = l.n / l.groups;
    int kk = l.size*l.size*l.c / l.groups;
    float *temp = (float*)malloc(sizeof(float)*m*kk);
    int cnt=0;
    for(i = 0; i<l.out_c/8; i++){
        for(j = 0; j<l.c; j++){
            for(k = 0; k<8; k++){
                for(w=0;w<9;w++) 
                    temp[cnt++] = l.weights[9*l.c*8*i + 9*l.c*k + 9*j + w];
            }
        }
    }
    for(i = 0; i < m*kk; i++) l.weights[i] = temp[i];
    free(temp);
}

#ifdef HALF_MODE
void weight_reorder_half(convolutional_layer l)
{
    int i, j, k, w;
    int m = l.n / l.groups;
    int kk = l.size*l.size*l.c / l.groups;
    cl_half *temp = (cl_half*)malloc(sizeof(cl_half)*m*kk);
    printf("allocation done!!!!!!!\n");
    int cnt=0;
    for(i = 0; i<l.out_c/8; i++){
        for(j = 0; j<l.c; j++){
            for(k = 0; k<8; k++){
                for(w=0;w<9;w++) 
                    temp[cnt++] = l.weights_half[9*l.c*8*i + 9*l.c*k + 9*j + w];
            }
        }
    }
    
    for(i = 0; i < m*kk; i++) l.weights_half[i] = temp[i];
    free(temp);
}
#endif // HALF_MODE

cl_mem create_mo_img(network *netp, int img_size, int uchar_im)
{
    int i;
    cl_mem mo_img;
    network net = *netp;

    if(uchar_im){
        mo_img = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(unsigned char) * img_size, NULL);
        unsigned char *im_copy = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
        for(i=0; i<img_size; i++)
        {  
            int tmp = net.input[i]*255;
            if(tmp < 0)
                tmp = 0;
            else if(tmp > 255)
                tmp = 255;
            im_copy[i] = (unsigned char)(tmp);
        }
        cl_memcpy_to_device(mo_img, im_copy, sizeof(unsigned char) * img_size);
        free(im_copy);
    }
    else{
        mo_img = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(float) * img_size, NULL);
        cl_memcpy_to_device(mo_img, net.input, sizeof(float) * img_size);
    }

    return mo_img;
}

void forward_network_ocl(network *netp)
{

#ifdef HALF_MODE
    forward_network_ocl_half(netp);
    return;
#endif

#ifdef FIXED_MODE
    forward_network_ocl_fixed(netp);
    return;
#endif

// FLOAT MODE
    network net = *netp;
    int i, c;
    double gpu_time = 0;
    double cpu_time = 0;

// config
    int convmax_0 = 0;
    int convmax_2 = 0;
    int uchar_im = 1;

    cl_mem mo_img, mo_buf_0, mo_buf_1;
    cl_mem *pmo_in, *pmo_out, *tmp;

    int img_size = find_input_size(net.layers[0]);
    int sz_out = find_max_out_size(netp);

    mo_img = create_mo_img(netp, img_size, uchar_im);
    mo_buf_0 = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(float) * sz_out, NULL);
    mo_buf_1 = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(float) * sz_out, NULL);

    if(VEC > 1){
        weight_reorder(net.layers[12]);
        weight_reorder(net.layers[13]);
    }

    int count = 0;

    char buf[256];
    for(c = 0; c < NUM_LOOP; c++){
        count = 0;
        if(c == 0)
            log_sw = 1;
        for(i = 0; i < net.n; ++i){
            net.index = i;
            layer l = net.layers[i];
            if(l.delta) fill_cpu(l.outputs * l.batch, 0, l.delta, 1);

            if (count == 0)         { pmo_in = &mo_img;   pmo_out = &mo_buf_0; }
            else if (count%2 == 0)  { pmo_in = &mo_buf_1; pmo_out = &mo_buf_0; }
            else                { pmo_in = &mo_buf_0; pmo_out = &mo_buf_1; }

            if(c==0){
                sprintf(buf, "\nlayer %d:", i);
                define_log(buf);
            }
            double time=what_time_is_it_now();
            switch(i){
                case 0:
                    if(convmax_0){
                        conv_ocl_float(l, net, pmo_in, pmo_out, -3);
                        i = 1;
                    }
                    else if(uchar_im)
                        conv_ocl_float(l, net, pmo_in, pmo_out, -2);
                    else
                        conv_ocl_float(l, net, pmo_in, pmo_out, -1);
                    break;
                case 2:
                    if(convmax_2){
                        conv_ocl_float(l, net, pmo_in, pmo_out, -4);
                        i = 3;
                        break;
                    }
                case 4:
                case 6:
                case 8:
                case 10:
                    conv_ocl_float(l, net, pmo_in, pmo_out, -1);
                    break;
                case 12:
                case 13:
                    conv_ocl_float(l, net, pmo_in, pmo_out, VEC);
                    break;
                case 14:
                    forward_convolutional_layer(l, net);
                    break;
                case 15:
                    forward_region_layer(l, net);
                    break;
                default:
                    maxpool_ocl(l, net, pmo_in, pmo_out);
            }

            if(i < 14)  gpu_time += what_time_is_it_now()-time;
            else        cpu_time += what_time_is_it_now()-time;

            if(i > 12){
                if(i == 13){
                    int output_size= l.out_w*l.out_h*l.out_c;
                    cl_memcpy_from_device(l.output, *pmo_out, sizeof(float) * output_size);
                }
                net.input = l.output;
                if(l.truth) net.truth = l.output;
            }
            count++;
        }

        if(c == 0)
            log_sw =0;
    }

    clFreeMemobj(mo_img);
    clFreeMemobj(mo_buf_0);
    clFreeMemobj(mo_buf_1);


    printf("Number of Class is : %d\n", NUM_LOOP);
    sprintf(buf, "\nNumber of Class is : %d", NUM_LOOP);
    define_log(buf);

    printf("GPU time %f seconds.\n\n", gpu_time);
    sprintf(buf, "GPU time %f seconds.", gpu_time);
    define_log(buf);

    printf("CPU time %f seconds.\n\n", cpu_time);
    sprintf(buf,"CPU time %f seconds.", cpu_time);
    define_log(buf);

    printf("Each GPU time %f seconds.\n\n", gpu_time/NUM_LOOP);
    sprintf(buf,"Each GPU time %f seconds.", gpu_time/NUM_LOOP);
    define_log(buf);

    calc_network_cost(netp);
}

#ifdef HALF_MODE
void forward_network_ocl_half(network *netp)
{
    network net = *netp;
    int i, c;
    double gpu_time = 0; 
    double cpu_time = 0;
    cl_mem mo_img, mo_buf_0, mo_buf_1;
    cl_mem *pmo_in, *pmo_out;

    int sz_out = 0;
    int img_size = net.layers[0].c * net.layers[0].w * net.layers[0].h;

    for(i = 0; i < net.n; i++){
        int n2 = net.layers[i].out_c * net.layers[i].out_w * net.layers[i].out_h;
        if( n2 > sz_out) sz_out = n2;
    }

    mo_img = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(cl_half) * img_size, NULL);
    cl_half *copy_input = malloc(sizeof(cl_half) * img_size);
    do_conversion_f_to_h(copy_input, net.input, img_size);
    cl_memcpy_to_device(mo_img, copy_input, sizeof(cl_half) * img_size);

    mo_buf_0 = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(cl_half) * sz_out, NULL);
    mo_buf_1 = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(cl_half) * sz_out, NULL);

    cl_half *output_cl_half = malloc(sizeof(cl_half) * sz_out);

    if(VEC > 1){
        weight_reorder_half(net.layers[12]);
        weight_reorder_half(net.layers[13]);
    }

    
    char buf[256];
    for(c = 0; c < NUM_LOOP; c++){
        printf("Class : %d\n", c);
        for(i = 0; i < net.n; ++i){
            net.index = i;
            layer l = net.layers[i];
            if(l.delta) fill_cpu(l.outputs * l.batch, 0, l.delta, 1);

            if (i == 0)         { pmo_in = &mo_img;   pmo_out = &mo_buf_0; }
            else if (i%2 == 0)  { pmo_in = &mo_buf_1; pmo_out = &mo_buf_0; }
            else                { pmo_in = &mo_buf_0; pmo_out = &mo_buf_1; }

            sprintf(buf, "\nlayer %d:", i);
            define_log(buf);

            double time=what_time_is_it_now();
            switch(i){
                case 0:
                case 2:
                case 4:
                case 6:
                case 8:
                case 10:
                    conv_ocl_half(l, net, pmo_in, pmo_out, -1);
                    break;
                case 12:
                case 13:
                    conv_ocl_half(l, net, pmo_in, pmo_out, VEC);
                    break;
                case 14:
                    forward_convolutional_layer(l, net);
                    break;
                case 15:
                    forward_region_layer(l, net);
                    break;
                default:
                    maxpool_ocl(l, net, pmo_in, pmo_out);
            }

            if(i < 14)  gpu_time += what_time_is_it_now()-time;
            else        cpu_time += what_time_is_it_now()-time;


            if(i > 12){
                if(i == 13){
                    int output_size= l.out_w*l.out_h*l.out_c;
                    cl_memcpy_from_device(output_cl_half, mo_buf_1, sizeof(cl_half) * output_size);    
                    do_conversion_h_to_f(l.output, output_cl_half, output_size);
                }
                net.input = l.output;
                if(l.truth) net.truth = l.output;
            }
        }
    }

    clFreeMemobj(mo_img);
    clFreeMemobj(mo_buf_0);
    clFreeMemobj(mo_buf_1);

    free(copy_input);
    free(output_cl_half); 
    printf("Number of Class is : %d\n", NUM_LOOP);
    printf("GPU time %f seconds.\n\n", gpu_time);
    printf("CPU time %f seconds.\n\n", cpu_time);

    printf("Each GPU time %f seconds.\n\n", gpu_time/NUM_LOOP);

    calc_network_cost(netp);    
}
#endif

#ifdef FIXED_MODE
void forward_network_ocl_fixed(network *netp)
{
    network net = *netp;
    int i, c;
    double gpu_time = 0;
    double cpu_time = 0;

    int convmax_0 = 0;
    int convmax_2 = 0;
    int uchar_im = 0;

    cl_mem mo_img, mo_buf_0, mo_buf_1;
    cl_mem *pmo_in, *pmo_out, *tmp;

    int img_size = find_input_size(net.layers[0]);
    int sz_out = find_max_out_size(netp);

    mo_img = create_mo_img(netp, img_size, uchar_im);

    mo_buf_0 = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(float) * sz_out, NULL);
    mo_buf_1 = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(float) * sz_out, NULL);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        int m = l.n / l.groups;
        int k = l.size*l.size*l.c / l.groups;
        if(l.weights_fixed){
            clSetupWmem(i, m, k, l.weights_fixed, l.biases, l.rolling_mean, l.rolling_variance, l.scales);
        }
    }

    if(VEC > 1){
        weight_reorder(net.layers[12]);
        weight_reorder(net.layers[13]);
    }

    int count = 0;

    char buf[256];
    for(c = 0; c < NUM_LOOP; c++){
        count = 0;
        if(c == 0)
            log_sw = 1;
        for(i = 0; i < net.n; ++i){
            net.index = i;
            layer l = net.layers[i];
            if(l.delta) fill_cpu(l.outputs * l.batch, 0, l.delta, 1);

            if (count == 0)         { pmo_in = &mo_img;   pmo_out = &mo_buf_0; }
            else if (count%2 == 0)  { pmo_in = &mo_buf_1; pmo_out = &mo_buf_0; }
            else                { pmo_in = &mo_buf_0; pmo_out = &mo_buf_1; }

            if(c==0){
                sprintf(buf, "\nlayer %d:", i);
                define_log(buf);
            }
            double time=what_time_is_it_now();
            switch(i){
                case 0:
                    conv_ocl_fixed(l, net, pmo_in, pmo_out, i);
                    break;
                case 2:
                case 4:
                case 6:
                case 8:
                case 10:
                    conv_ocl_fixed(l, net, pmo_in, pmo_out, i);
                    break;
                case 12:
                case 13:
                    conv_ocl_fixed(l, net, pmo_in, pmo_out, i);
                    break;
                case 14:
                    forward_convolutional_layer(l, net);
                    break;
                case 15:
                    forward_region_layer(l, net);
                    break;
                default:
                    maxpool_ocl(l, net, pmo_in, pmo_out);
            }

            if(i < 14)  gpu_time += what_time_is_it_now()-time;
            else        cpu_time += what_time_is_it_now()-time;

            if(i > 12){
                if(i == 13){
                    int output_size= l.out_w*l.out_h*l.out_c;
                    cl_memcpy_from_device(l.output, *pmo_out, sizeof(float) * output_size);
                }
                net.input = l.output;
                if(l.truth) net.truth = l.output;
            }
            count++;
        }

        if(c == 0)
            log_sw =0;

    }

    clFreeMemobj(mo_img);
    clFreeMemobj(mo_buf_0);
    clFreeMemobj(mo_buf_1);
    clReleaseWmem();

    printf("Number of Class is : %d\n", NUM_LOOP);
    sprintf(buf, "\nNumber of Class is : %d", NUM_LOOP);
    define_log(buf);

    printf("GPU time %f seconds.\n\n", gpu_time);
    sprintf(buf, "GPU time %f seconds.", gpu_time);
    define_log(buf);

    printf("CPU time %f seconds.\n\n", cpu_time);
    sprintf(buf,"CPU time %f seconds.", cpu_time);
    define_log(buf);

    printf("Each GPU time %f seconds.\n\n", gpu_time/NUM_LOOP);
    sprintf(buf,"Each GPU time %f seconds.", gpu_time/NUM_LOOP);
    define_log(buf);

    calc_network_cost(netp);
}
#endif // FIXED_MODE

#endif // OPENCL