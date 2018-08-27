//-------------------------------------------------------------
//
//  PROGRAM: Blocked Matrix Multipliplication kernel
//
//  PURPOSE: Computes an element of the proudct matrix
//
//              C = A * B
//
//           Using the well known blocked algorithm.  
//
//           To derive this algorithm, start with the naive
//           triply nested loop algorithm with a dot product 
//           for each element of C.  Decompose each loop 
//           into blocks of size blcksz.  This gives you 6
//           nested loops with three loops over blocks
//           and three loops over indices inside the blocks.
// 
//           Rearrange the loops to put the 3 loops over blocks 
//           at the outermost loops of the loop nest.  You'll
//           see that the three "inner" loops are just the 
//           regular matrix product between blocks.
//
//           The algorithms is simple.  Keeping all the indices
//           straight is not.  We will use the following 
//           conventions:
//
//             i,j,k            ... indices of full, global matrices 
//             Iblk, Jblk, Kblk ... indices of matrix blocks
//             iloc, jloc, kloc ... indices inside blocks
//                 
//  HISTORY: Written by Tim Mattson, November 2013 
//           Updated by Simon McIntosh-Smith, August 2014 
//
//  LICENSE: This work is licensed under the Creative Commons
//           Attribution 4.0 International License.
//           To view a copy of this license, visit
//           http://creativecommons.org/licenses/by/4.0/
//           or send a letter to:
//              Creative Commons,
//              444 Castro Street, Suite 900,
//              Mountain View, California, 94041, USA.
//
//-------------------------------------------------------------
#define LOCAL_WIDTH 13
#define LOCAL_HEIGHT 13
#define LOCAL_DEPTH 16
//#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Conv3(__global half* imageA, __global half* imageB, __global half* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, 
				   __global const half* mean, __global const half* variance,
				   __global const half* scales, __global const half* biases)
{
	__local half localImage[15][15];
	__local half localFilter[LOCAL_DEPTH][9];
	// 0 : z, 1: y, 2 : x
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	size_t g3 = get_global_size(2);

	int ii = 0;
	int i = 0;
	int j = 0;
	int k = 0;
	half myImage[3][3];
	half ans[LOCAL_DEPTH] = {0.0}; 
	// check if initialization chere is possible
	half mine;
	int my_idx, my_idx2;
	__local int pad_below, pad_above, pad_right, pad_left;
	__local half m2[LOCAL_DEPTH];
	__local half v2[LOCAL_DEPTH];
	__local half s2[LOCAL_DEPTH];
	__local half b2[LOCAL_DEPTH];
	int idx_1d = local_y * 13 + local_x;
	int chunk_num = idx_1d / 9;
	int inner_i;
	

	if(local_x == 0 && local_y == 0){
			pad_above = 0;
			pad_below = 0;
			pad_left = 0;
			pad_right = 0;
	}
	/**************** initialization ******************/
	//for(i = 0; i<LOCAL_DEPTH; i++) ans[i] = 0.0;


	/**************** padding *********************/
	if(global_y == 0 && local_x == 0 && local_y == 0){
		for(k = 0; k<15; k++) localImage[0][k] = 0.0;
		pad_above = 1;
	}
	
	else if(global_y == inputHeight -1 && local_x == 0 && local_y == LOCAL_HEIGHT - 1){
		for(k = 0; k<15; k++) localImage[14][k] = 0.0;
	//	if(global_x < 20 && global_y < 20 && global_z == 0)
	//	printf("padding below, global : %d %d %d, local : %d %d\n", global_z, global_y, global_x, local_x, local_y);
		pad_below = 1;
	}
	
	if(global_x == 0 && local_x == 0 && local_y == 0 ){
		for(k = 0; k<15; k++) localImage[k][0] = 0.0;
		pad_left = 1;
	}
	else if(global_x == inputWidth -1 && local_x == LOCAL_WIDTH -1 && local_y == LOCAL_HEIGHT - 1){
		for(k = 0; k<15; k++) localImage[k][14] = 0.0;
		pad_right = 1;
	}
	/************************************************/
	barrier(CLK_LOCAL_MEM_FENCE);

	for( i = 0; i< inputChannel; i++){
			
		my_idx = i*inputWidth*inputHeight+global_y*inputWidth+global_x;
		mine = imageA[my_idx];
		localImage[local_y+1][local_x+1] = mine;

	    if(local_x == 0 && pad_left != 1){
			localImage[local_y+1][0] = imageA[my_idx - 1];
		}
		else if(local_x == LOCAL_WIDTH - 1 && pad_right != 1){
			localImage[local_y+1][14] = imageA[my_idx + 1];
		}
		    
		if(local_y == 0 && pad_above != 1){
		   	localImage[0][local_x+1] = imageA[my_idx - inputWidth];
		}
		else if(local_y == LOCAL_HEIGHT -1 && pad_below != 1){
		  	localImage[local_y+2][local_x+1] = imageA[my_idx + inputWidth];
		}

		if(local_x == 0 && local_y == 0 && pad_left != 1 && pad_above != 1){
		   	localImage[0][0] = imageA[my_idx - inputWidth - 1];
		}
		else if(local_x == 0 && local_y == LOCAL_HEIGHT -1 && pad_left != 1 && pad_below != 1){
		   	localImage[14][0] = imageA[my_idx + inputWidth -1];
		}
		    
		if(local_x == LOCAL_WIDTH - 1 && local_y == 0 && pad_above != 1 && pad_right != 1){
		  	localImage[0][14] = imageA[my_idx - inputWidth + 1];
		}
		else if(local_x == LOCAL_WIDTH - 1 && local_y == LOCAL_HEIGHT - 1 && pad_below != 1 && pad_right != 1){
			localImage[14][14] = imageA[my_idx + inputWidth + 1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		///////////////////////////// copy filter /////////////////////////////////////////
		if(idx_1d < 9*LOCAL_DEPTH){
			localFilter[chunk_num][idx_1d%9] = imageB[(LOCAL_DEPTH*global_z+chunk_num)*9*inputChannel + i*9 + idx_1d%9];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		/////////////////////////////////////////////////////////////////////////////////////
		for(j = 0; j<3; j++){
			for(k = 0; k<3; k++){
				myImage[j][k] = localImage[local_y+j][local_x+k];
			}
		}
		///////////////////////////////////// convolution starts /////////////////////////
		for(inner_i = 0; inner_i < LOCAL_DEPTH; inner_i++){
			for (j = 0 ; j<3; j++){
				for(k = 0; k <3 ; k++){
					ans[inner_i] += (localFilter[inner_i][3*j+k] * myImage[j][k]);
//					ans[inner_i] += (localFilter[inner_i][3*j+k] * localImage[local_y+j][local_x+k]);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	my_idx2 = LOCAL_DEPTH*global_z + idx_1d%LOCAL_DEPTH;
	if(idx_1d < LOCAL_DEPTH) m2[idx_1d] = mean[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*2) v2[idx_1d%LOCAL_DEPTH] = variance[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*3) s2[idx_1d%LOCAL_DEPTH] = scales[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*4) b2[idx_1d%LOCAL_DEPTH] = biases[my_idx2];

	barrier(CLK_LOCAL_MEM_FENCE);

	for(i = 0; i<LOCAL_DEPTH; i++){
//		ans[i] = (ans[i] - m2[i]) / (sqrt(v2[i]) + .000001f);
		ans[i] = (ans[i] - m2[i]) / (v2[i] + .000001f);
		ans[i] = ans[i]*s2[i] + b2[i];

		if(ans[i] < 0) ans[i] *= 0.1f;
		imageC[inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x] = ans[i];
	}

/*
	__local half lidx;
	lidx = 0;
	for(i = 0; i<1000; i++)
		for(j = 0; j<1000; j++)
			lidx = lidx + .00001f;
*/
/*
	__private half pidx;
	pidx = 0;
	for(i = 0; i<1000; i++)
		for(j = 0; j<1000; j++)
			pidx= pidx + .00001f;
*/
}

__kernel void Pool2(__global half* imageA, __global half* imageC)
{
	__local half localImage[13][13];
	
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);

	size_t local_x = get_local_id(0);
	size_t local_y = get_local_id(1);
	
	size_t g1 = get_global_size(0);
	size_t g2 = get_global_size(1);
	size_t g3 = get_global_size(2);
	int i, j, k;
	half v;
	int my_idx = (13*13*global_z) + (13*global_y) + global_x;
	localImage[local_y][local_x] = imageC[my_idx];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	v = localImage[local_y][local_x];
	
	if(local_x == 12 && local_y == 12){}
	else if(local_x < 12 && local_y < 12)
	{
		for(i = 0; i <2; i++){
			for(j = 0; j <2; j++){
				if(localImage[local_y+i][local_x+j] > v) v = localImage[local_y+i][local_x+j];
			}
		}
	}
	else if(local_x == 12){
		if(localImage[local_y + 1][local_x] > v) v = localImage[local_y + 1][local_x];
	}
	else if(local_y == 12){
		if(localImage[local_y][local_x + 1] > v) v = localImage[local_y][local_x + 1];
	}
	
	imageA[my_idx] = v;

}


__kernel void Pool(__global half* imageA, __global half* imageC)
{
	__local half localImage[2][26];
	
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	
	size_t g1 = get_global_size(0);
	size_t g2 = get_global_size(1);
	size_t g3 = get_global_size(2);

	int ii = 0;
	int i = 0;
	int j = 0;
	int k = 0;
	half ans;
	half mine;
	int my_idx;
	int idx;

	my_idx = global_z*g1*g2+global_y*g1+global_x;
	localImage[local_y][local_x] = imageC[my_idx];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(local_x %2 == 0 && local_y %2 == 0){
		ans = localImage[local_y][local_x];
		for(i = 0; i <2; i++){
			for(j = 0; j <2; j++){
				if(ans < localImage[local_y + i][local_x + j]){
					ans = localImage[local_y + i][local_x + j];
				}
			}
		}
		idx = global_z*(g1/2)*(g2/2) + global_y/2* g1/2 + global_x/2;
		imageA[idx] = ans;
	}		
}

__kernel void Move(__global half* imageA, __global half* imageC, const int filter_num){
	
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);

	size_t g1 = get_global_size(0);
	size_t g2 = get_global_size(1);
	size_t g3 = get_global_size(2);

	int ii = 0;

	int my_idx;

	for(ii = 0 ; ii < filter_num/g3; ii++){
		my_idx = (ii*g3+global_z)*g1*g2 + global_y*g1 + global_x;
		imageA[my_idx] = imageC[my_idx];
	}
}

__kernel void Test_cl(__global float* input, __global float* output, __local float* arr)
{

	int i, j;

	for(j=0; j<100; j++)
		arr[j] = .00001f*j;

	for(i=0; i<100000; i++)
		for(j=0; j<100; j++)
			arr[j] += arr[j];
}
