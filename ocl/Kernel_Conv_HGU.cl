/**
  * @file       Kernel_Conv_HGU.cl
  * @author     Yechan, Jinju, Daewoo
  * @date       2018-12-03
  * @email      21400067@handong.edu
  * @brief      OpenCL kernel for                   \n
                convolution & batchnormalization    \n
                of 'float' version                  \n
**/




// function Conv parameters
// 	imageA : image
// 	imageB : weight
// 	imageC : output
__kernel void Conv(__global float* imageA, __global unsigned char* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, 
				   __global const float* mean, __global const float* variance,
				   __global const float* scales, __global const float* biases,
				   const float wrmin, const float step)
{
	__local float localImage[15][15];
	__local float localFilter[LOCAL_DEPTH][9];

	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	
	int ii = 0;
	int i = 0;
	int j = 0;
	int k = 0;
	float myImage[3][3];
	float ans[LOCAL_DEPTH] = {0.0}; 
	float px = 0.0; 
	float pwrmin = wrmin;
	float pstep = step;
	int my_idx, my_idx2;
	__local int pad_below, pad_above, pad_right, pad_left;
	__local float m2[LOCAL_DEPTH];
	__local float v2[LOCAL_DEPTH];
	__local float s2[LOCAL_DEPTH];
	__local float b2[LOCAL_DEPTH];
	int idx_1d = local_y * 13 + local_x;
	int chunk_num = idx_1d / 9;
	int inner_i;
	int ftbias = (LOCAL_DEPTH*global_z+chunk_num)*9;
	int ichstep = filter_num*9;

	if(local_x == 0 && local_y == 0){
			pad_above = 0;
			pad_below = 0;
			pad_left = 0;
			pad_right = 0;
	}
	

	/**************** padding *********************/
	if(global_y == 0 && local_x == 0 && local_y == 0){		//enable padding at top
		for(k = 0; k<15; k++) localImage[0][k] = 0.f;
		pad_above = 1;
	}
	
	else if(global_y == inputHeight -1 && local_x == 0 && local_y == LOCAL_HEIGHT - 1){		//enable padding at bottom
		for(k = 0; k<15; k++) localImage[14][k] = 0.f;
		pad_below = 1;
	}
	
	if(global_x == 0 && local_x == 0 && local_y == 0 ){		//enable padding at left
		for(k = 0; k<15; k++) localImage[k][0] = 0.f;
		pad_left = 1;
	}
	else if(global_x == inputWidth -1 && local_x == LOCAL_WIDTH -1 && local_y == LOCAL_HEIGHT - 1){		//enable padding at right
		for(k = 0; k<15; k++) localImage[k][14] = 0.f;
		pad_right = 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/********************copy image global->local****************************/
	for( i = 0; i< inputChannel; i++){
		my_idx = i*inputWidth*inputHeight + (global_y/13)*13*inputWidth + (global_x/13)*13;
		if(pad_above)
		{
			if(pad_left)		//case of padding at top & left
			{
				for(j=local_x+local_y*13; j < 14*14; j+=13*13){
					localImage[j/14+1][j%14+1] = imageA[my_idx + (j/14)*inputWidth + j%14];
				}
			}
			else if(pad_right)  //case of padding at top & right
			{
				for(j=local_x+local_y*13; j < 14*14; j+=13*13){
					localImage[j/14+1][j%14] = imageA[my_idx + (j/14)*inputWidth + j%14 - 1];
				}
			}
			else				//case of padding at top
			{
				for(j=local_x+local_y*13; j < 14*15; j+=13*13){
					localImage[j/15+1][j%15] = imageA[my_idx + (j/15)*inputWidth + j%15 - 1];
				}
			}
		}
		else if(pad_below)
		{
			if(pad_left)		//case of padding at bottom & left
			{
				for(j=local_x+local_y*13; j < 14*14; j+=13*13){
					localImage[j/14][j%14+1] = imageA[my_idx + (j/14 - 1)*inputWidth + j%14];
				}
			}
			else if(pad_right)	//case of padding at bottom & right
			{
				for(j=local_x+local_y*13; j < 14*14; j+=13*13){
					localImage[j/14][j%14] = imageA[my_idx + (j/14 - 1)*inputWidth + j%14 - 1];
				}
			}
			else				//case of padding at bottom
			{
				for(j=local_x+local_y*13; j < 15*14; j+=13*13){
					localImage[j/15][j%15] = imageA[my_idx + (j/15 - 1)*inputWidth + j%15 - 1];
				}
			}
		}
		else{
			if(pad_left)		//case of padding at left
			{
				for(j=local_x+local_y*13; j < 15*14; j+=13*13){
					localImage[j/14][j%14+1] = imageA[my_idx + (j/14 - 1)*inputWidth + j%14];
				}
			}
			else if(pad_right)	//case of padding at right
			{
				for(j=local_x+local_y*13; j < 15*14; j+=13*13){
					localImage[j/14][j%14] = imageA[my_idx + (j/14 - 1)*inputWidth + j%14 - 1];
				}
			}
			else				//case of no padding
			{
				for(j=local_x+local_y*13; j < 15*15; j+=13*13){
					localImage[j/15][j%15] = imageA[my_idx + (j/15 - 1)*inputWidth + j%15 - 1];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
			
		/********************copy filter global->local****************************/
		
		if(idx_1d < 9*LOCAL_DEPTH){
			localFilter[chunk_num][idx_1d%9] = imageB[i*ichstep + ftbias + idx_1d%9];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		/********************copy image local->private****************************/
		for(j = 0; j<3; j++){
			for(k = 0; k<3; k++){
				myImage[j][k] = localImage[local_y+j][local_x+k];
				px += myImage[j][k];
			}
		}
		/********************convolution****************************/
		for(inner_i = 0; inner_i < LOCAL_DEPTH; inner_i++){
			for (j = 0 ; j<3; j++){
				for(k = 0; k <3 ; k++){
					ans[inner_i] += (localFilter[inner_i][3*j+k]* myImage[j][k]);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/********************copy batchnorm global->local****************************/
	my_idx2 = LOCAL_DEPTH*global_z + idx_1d%LOCAL_DEPTH;
	if(idx_1d < LOCAL_DEPTH) m2[idx_1d] = mean[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*2) v2[idx_1d%LOCAL_DEPTH] = variance[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*3) s2[idx_1d%LOCAL_DEPTH] = scales[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*4) b2[idx_1d%LOCAL_DEPTH] = biases[my_idx2];

	barrier(CLK_LOCAL_MEM_FENCE);

	/********************calculate batchnorm****************************/
	for(i = 0; i<LOCAL_DEPTH; i++){
		ans[i] = (ans[i] + px*pwrmin/pstep)*pstep;
		ans[i] = (ans[i] - m2[i]) / (sqrt(v2[i]) + .000001f); // can reduce more, initialize to negative mean value
		ans[i] = ans[i]*s2[i] + b2[i];

	/********************copy output image private->global****************************/
		if(ans[i] < 0) ans[i] *= 0.1f;
		imageC[inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x] = ans[i];
	}
}
