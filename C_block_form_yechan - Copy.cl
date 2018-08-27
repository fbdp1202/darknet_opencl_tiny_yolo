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

__kernel void fc(__global float* imageA, __global float* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num
				   )
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);

	int i = 0;

	float ans = 0.0;
	float mine;
	__local float weight;
	int my_idx;
	
	for(i = 0; i < inputChannel; i++){
		my_idx = i*inputWidth*inputHeight + inputWidth*global_y + global_x;
		mine = imageA[my_idx];
		
		if(global_x == 0 && global_y == 0)
			weight = imageB[global_z*inputChannel+i];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		ans += (mine * weight);
	
	}
	
		
	imageC[inputWidth*inputHeight*global_z + inputWidth*global_y + global_x] = ans;
}

__kernel void Conv2(__global float* imageA, __global float* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, 
				   __global const float* mean, __global const float* variance,
				   __global const float* scales, __global const float* biases, const int norm
				   )
{
	__local float localImage[15][15];
	__local float localFilter[3][3];
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

	float ans = 0.0;
	float mine;
	int my_idx;
	__local int pad_below, pad_above, pad_right, pad_left;
	__local float m2;
	__local float v2;
	__local float s2;
	__local float b2;
	
	
	if(local_x == 0 && local_y == 0){
			pad_above = 0;
			pad_below = 0;
			pad_left = 0;
			pad_right = 0;
	}
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
		    /*********************************************/
		    barrier(CLK_LOCAL_MEM_FENCE);
	
	for(ii = 0 ; ii < filter_num/g3; ii++){
		ans = 0.0;
		
		for( i = 0; i< inputChannel; i++){
			
			my_idx = i*inputWidth*inputHeight+global_y*inputWidth+global_x;
			mine = imageA[my_idx];
			localImage[local_y+1][local_x+1] = mine;
						    
		    if(local_x == 0 && pad_left != 1){
				localImage[local_y+1][0] = imageA[my_idx - 1];	
			}
			else if(local_x == LOCAL_WIDTH - 1 && pad_right != 1)
				localImage[local_y+1][14] = imageA[my_idx + 1];
		    
		    if(local_y == 0 && pad_above != 1)
		    	localImage[0][local_x+1] = imageA[my_idx - inputWidth];
		    else if(local_y == LOCAL_HEIGHT -1 && pad_below != 1)
		    	localImage[local_y+2][local_x+1] = imageA[my_idx + inputWidth];
		    	
		    if(local_x == 0 && local_y == 0 && pad_left != 1 && pad_above != 1)
		    	localImage[0][0] = imageA[my_idx - inputWidth - 1];
		    else if(local_x == 0 && local_y == LOCAL_HEIGHT -1 && pad_left != 1 && pad_below != 1)
		    	localImage[14][0] = imageA[my_idx + inputWidth -1];
		    
		    if(local_x == LOCAL_WIDTH - 1 && local_y == 0 && pad_above != 1 && pad_right != 1)
		    	localImage[0][14] = imageA[my_idx - inputWidth + 1];
			else if(local_x == LOCAL_WIDTH - 1 && local_y == LOCAL_HEIGHT - 1 && pad_below != 1 && pad_right != 1)
				localImage[14][14] = imageA[my_idx + inputWidth + 1];
			
			barrier(CLK_LOCAL_MEM_FENCE);
			
			///////////////////////////// copy filter /////////////////////////////////////////
			if(local_x < 3 && local_y <3){
				localFilter[local_y][local_x] = imageB[(ii*g3+global_z)*9*inputChannel +9*i+(3*local_y) + local_x];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			
			///////////////////////////////////// convolution starts /////////////////////////
			for (j = 0 ; j<3; j++){
				for(k = 0; k <3 ; k++){

					ans+= (localFilter[j][k] * localImage[local_y+j][local_x+k]);
					
				}
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
		
		}


		if(local_x == 0 && local_y == 0){
			if(norm == 1){
				m2 = mean[ii*g3 + global_z];
				v2 = variance[ii*g3 + global_z];
				s2 = scales[ii*g3 + global_z];
			}
			b2 = biases[ii*g3 + global_z];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if(norm == 1){
			ans = (ans - m2) / (sqrt(v2) + .000001f);
			ans = ans*s2;
		}
		ans = ans + b2;
		if(ans < 0) ans *= 0.1f;
		imageC[inputWidth*inputHeight*(ii*g3+global_z) + inputWidth*global_y + global_x] = ans;
	}
}


__kernel void Conv(__global float* imageA, __global float* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, __global const float* mean, __global const float* variance,
				   __global const float* scales, __global const float* biases, const int norm)
{
		__local float localImage[15][15];
	__local float localFilter[3][3];
	// 0 : z, 1: y, 2 : x
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	size_t g1 = get_global_size(0);
	size_t g3 = get_global_size(2);

	int ii = 0;
	int i = 0;
	int j = 0;
	int k = 0;

	float ans = 0.0;
	float mine;
	int my_idx;
	int idx2, idx3, idx4;
	__local int pad_below, pad_above, pad_right, pad_left;
	__local float m2;
	__local float v2;
	__local float s2;
	__local float b2;
	
	
	if(local_x == 0 && local_y == 0){
			pad_above = 0;
			pad_below = 0;
			pad_left = 0;
			pad_right = 0;
	}
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
		    /*********************************************/
		    barrier(CLK_LOCAL_MEM_FENCE);
	
	
		ans = 0.0;
		
		for( i = 0; i< inputChannel; i++){
			
			my_idx = i*inputWidth*inputHeight+global_y*inputWidth+global_x;
			mine = imageA[my_idx];
			localImage[local_y+1][local_x+1] = mine;
						    
		    if(local_x == 0 && pad_left != 1){
				localImage[local_y+1][0] = imageA[my_idx - 1];	
			}
			else if(local_x == LOCAL_WIDTH - 1 && pad_right != 1)
				localImage[local_y+1][14] = imageA[my_idx + 1];
		    
		    if(local_y == 0 && pad_above != 1)
		    	localImage[0][local_x+1] = imageA[my_idx - inputWidth];
		    else if(local_y == LOCAL_HEIGHT -1 && pad_below != 1)
		    	localImage[local_y+2][local_x+1] = imageA[my_idx + inputWidth];
		    	
		    if(local_x == 0 && local_y == 0 && pad_left != 1 && pad_above != 1)
		    	localImage[0][0] = imageA[my_idx - inputWidth - 1];
		    else if(local_x == 0 && local_y == LOCAL_HEIGHT -1 && pad_left != 1 && pad_below != 1)
		    	localImage[14][0] = imageA[my_idx + inputWidth -1];
		    
		    if(local_x == LOCAL_WIDTH - 1 && local_y == 0 && pad_above != 1 && pad_right != 1)
		    	localImage[0][14] = imageA[my_idx - inputWidth + 1];
			else if(local_x == LOCAL_WIDTH - 1 && local_y == LOCAL_HEIGHT - 1 && pad_below != 1 && pad_right != 1)
				localImage[14][14] = imageA[my_idx + inputWidth + 1];
			
			barrier(CLK_LOCAL_MEM_FENCE);
			
			///////////////////////////// copy filter /////////////////////////////////////////
			if(local_x < 3 && local_y <3){
				if(g1 > inputWidth) idx2 = (global_z*2 + global_x/inputWidth)*9*inputChannel +9*i+(3*local_y) + local_x;
				else idx2 = global_z*9*inputChannel +9*i+(3*local_y) + local_x;
				localFilter[local_y][local_x] = imageB[idx2];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			
			///////////////////////////////////// convolution starts /////////////////////////
			for (j = 0 ; j<3; j++){
				for(k = 0; k <3 ; k++){

					ans+= (localFilter[j][k] * localImage[local_y+j][local_x+k]);
					
				}
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
		
		}


		if(local_x == 0 && local_y == 0){
			if(norm == 1){
				if(g1 >  inputWidth) idx4 = global_z*2 + global_x/inputWidth;
				else  idx4 = global_z;
				m2 = mean[idx4];
				v2 = variance[idx4];
				s2 = scales[idx4];
			}
			b2 = biases[idx4];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if(norm == 1){
			ans = (ans - m2) / (sqrt(v2) + .000001f);
			ans = ans*s2;
		}
		ans = ans + b2;
		if(ans < 0) ans *= 0.1f;
		if(g1 > inputWidth) idx3 = (global_z*2 + global_x/inputWidth)*inputWidth*inputHeight +inputWidth*global_y + global_x;
		else idx3 = global_z*inputWidth*inputHeight +inputWidth*global_y + global_x;
		imageC[idx3] = ans;
	
}


__kernel void Pool2(__global float* imageA, __global float* imageC)
{
	__local float localImage[13][13];
	
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);

	size_t local_x = get_local_id(0);
	size_t local_y = get_local_id(1);
	
	size_t g1 = get_global_size(0);
	size_t g2 = get_global_size(1);
	size_t g3 = get_global_size(2);
	int i, j, k;
	float v;
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


__kernel void Pool(__global float* imageA, __global float* imageC)
{
	__local float localImage[2][26];
	
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
	float ans;
	float mine;
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

__kernel void Move(__global float* imageA, __global float* imageC, const int filter_num){
	
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
