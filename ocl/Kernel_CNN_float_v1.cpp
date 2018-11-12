#define LOCAL_WIDTH 13
#define LOCAL_HEIGHT 13

#define LOCAL_DEPTH 16
#define LOCAL_DEPTH2 16

typedef union lptr16{
	__local float16 *vec;
	__local float *arr;
} lptr16;
typedef union lptr8{
	__local float8 *vec;
	__local float *arr;
} lptr8;
typedef union lptr4{
	__local float4 *vec;
	__local float *arr;
} lptr4;
typedef union lptr2{
	__local float2 *vec;
	__local float *arr;
} lptr2;


__kernel void Conv3_vec8(__global float8* imageA, __global float8* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, 
				   __global const float8* mean, __global const float8* variance,
				   __global const float8* scales, __global const float8* biases)
{
	__local float8 localImage[22];
	__local float8 localFilter[9];
	//__local float localFilter[8][9];
	// 0 : z, 1: y, 2 : x
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int i = 0;
	int j = 0;
	int k = 0;
	float myImage[3][3];
	float ans[LOCAL_DEPTH] = {0.0}; 
	// check if initialization here is possible
	float mine;
	int my_idx, my_idx2;
	__local float8 m2[LOCAL_DEPTH/8];
	__local float8 v2[LOCAL_DEPTH/8];
	__local float8 s2[LOCAL_DEPTH/8];
	__local float8 b2[LOCAL_DEPTH/8];
	int idx_1d = global_y * 13 + global_x;
	//int chunk_num = idx_1d / 9;
	int inner_i;
	int i_mod_8, my_i, my_j, my_i_from, my_j_from;
	lptr8 u_img, u_filt, u_m, u_v, u_s, u_b;

	for( i = 0; i< inputChannel; i++){
		i_mod_8 = i%8;
		 
		if(idx_1d < 22){
			my_idx = (int)(i/8)*169 + 21*i_mod_8 + idx_1d;
			
			localImage[idx_1d] = imageA[my_idx];
			
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		u_img.vec = localImage;
		u_img.arr = u_img.arr + i_mod_8;
		///////////////////////////// copy image to private ////////////////////////////////////////////////////////
		
		my_i_from = global_y - 1;
		my_j_from = global_x - 1;
		for(j = 0; j<3; j++){
			my_i = my_i_from + j;
			my_j = global_x - 1;
			for(k = 0; k<3; k++){
				my_j = my_j_from + k;
				if(my_i < 0 || my_j < 0 || my_i > 12 || my_j > 12) myImage[j][k] = 0.0f;
				else myImage[j][k] = u_img.arr[13* my_i + my_j];
			}
		}

		///////////////////////////// copy filter /////////////////////////////////////////
			
			////////////////////// scalar version ///////////////////////
			/*
			if(idx_1d < 9*LOCAL_DEPTH){
			localFilter[chunk_num][idx_1d%9] = vload_half((LOCAL_DEPTH*global_z+chunk_num)*9*inputChannel + i*9 + idx_1d%9, imageB);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			*/
			////////////////////// vector version ///////////////////////
			
			if(idx_1d < 9){
				my_idx = (9*inputChannel*8*global_z + 72*i)/8 + idx_1d;
				localFilter[idx_1d] = imageB[my_idx];

			}
			barrier(CLK_LOCAL_MEM_FENCE);
			u_filt.vec = localFilter;
			
		///////////////////////////////////// convolution starts /////////////////////////
		for(inner_i = 0; inner_i < LOCAL_DEPTH; inner_i++){
			for (j = 0 ; j<3; j++){
				for(k = 0; k <3 ; k++){
					ans[inner_i] += (u_filt.arr[9 * inner_i + 3*j+k] * myImage[j][k]);
					//ans[inner_i] += (localFilter[inner_i][3*j+k] * myImage[j][k]);
				}
			}	
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	int ld_div_8 = LOCAL_DEPTH / 8; 
	my_idx2 = ld_div_8*global_z + idx_1d%(ld_div_8);
	if(idx_1d < ld_div_8) m2[idx_1d] = mean[my_idx2];
	else if(idx_1d < ld_div_8*2) v2[idx_1d%(ld_div_8)] = variance[my_idx2];
	else if(idx_1d < ld_div_8*3) s2[idx_1d%(ld_div_8)] = scales[my_idx2];
	else if(idx_1d < ld_div_8*4) b2[idx_1d%(ld_div_8)] = biases[my_idx2];

	u_m.vec = m2; 
	u_v.vec = v2;
	u_s.vec = s2;
	u_b.vec = b2;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(i = 0; i<LOCAL_DEPTH; i++){
		ans[i] = (ans[i] - u_m.arr[i]) / (sqrt(u_v.arr[i]) + .000001f);
		ans[i] = ans[i]*u_s.arr[i] + u_b.arr[i];

		if(ans[i] < 0) ans[i] *= 0.1f;
		imageC[inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x] = ans[i];
		//vstore_half(ans[i], inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x, imageC);
		
	}
	
}


__kernel void Conv3_vec4(__global float4* imageA, __global float4* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, 
				   __global const float4* mean, __global const float4* variance,
				   __global const float4* scales, __global const float4* biases)
{
	__local float4 localImage[43];
	__local float4 localFilter[18];
	//__local float localFilter[8][9];
	// 0 : z, 1: y, 2 : x
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int i = 0;
	int j = 0;
	int k = 0;
	float myImage[3][3];
	float ans[LOCAL_DEPTH] = {0.0}; 
	// check if initialization here is possible
	float mine;
	int my_idx, my_idx2;
	__local float4 m2[LOCAL_DEPTH/4];
	__local float4 v2[LOCAL_DEPTH/4];
	__local float4 s2[LOCAL_DEPTH/4];
	__local float4 b2[LOCAL_DEPTH/4];
	int idx_1d = global_y * 13 + global_x;
	//int chunk_num = idx_1d / 9;
	int inner_i;
	int i_mod_4, my_i, my_j, my_i_from, my_j_from;
	lptr4 u_img, u_filt, u_m, u_v, u_s, u_b;
	
	for( i = 0; i< inputChannel; i++){
		i_mod_4 = i%4;
		 
		if(idx_1d < 43){
			my_idx = (int)(i/4)*169 + 42*i_mod_4 + idx_1d;
			
			localImage[idx_1d] = imageA[my_idx];
			
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		u_img.vec = localImage;
		u_img.arr = u_img.arr + i_mod_4;
		///////////////////////////// copy image to private ////////////////////////////////////////////////////////
		
		my_i_from = global_y - 1;
		my_j_from = global_x - 1;
		for(j = 0; j<3; j++){
			my_i = my_i_from + j;
			my_j = global_x - 1;
			for(k = 0; k<3; k++){
				my_j = my_j_from + k;
				if(my_i < 0 || my_j < 0 || my_i > 12 || my_j > 12) myImage[j][k] = 0.0f;
				else myImage[j][k] = u_img.arr[13* my_i + my_j];
			}
		}

		///////////////////////////// copy filter /////////////////////////////////////////
			
			////////////////////// scalar version ///////////////////////
			/*
			if(idx_1d < 9*LOCAL_DEPTH){
			localFilter[chunk_num][idx_1d%9] = vload_half((LOCAL_DEPTH*global_z+chunk_num)*9*inputChannel + i*9 + idx_1d%9, imageB);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			*/
			////////////////////// vector version ///////////////////////
			
			if(idx_1d < 18){
				my_idx = (9*inputChannel*8*global_z + 72*i)/4 + idx_1d;
				localFilter[idx_1d] = imageB[my_idx];

			}
			barrier(CLK_LOCAL_MEM_FENCE);
			u_filt.vec = localFilter;
			
		///////////////////////////////////// convolution starts /////////////////////////
		for(inner_i = 0; inner_i < LOCAL_DEPTH; inner_i++){
			for (j = 0 ; j<3; j++){
				for(k = 0; k <3 ; k++){
					ans[inner_i] += (u_filt.arr[9 * inner_i + 3*j+k] * myImage[j][k]);
					//ans[inner_i] += (localFilter[inner_i][3*j+k] * myImage[j][k]);
				}
			}	
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	int ld_div_4 = LOCAL_DEPTH / 4; 
	my_idx2 = ld_div_4*global_z + idx_1d%(ld_div_4);
	if(idx_1d < ld_div_4) m2[idx_1d] = mean[my_idx2];
	else if(idx_1d < ld_div_4*2) v2[idx_1d%(ld_div_4)] = variance[my_idx2];
	else if(idx_1d < ld_div_4*3) s2[idx_1d%(ld_div_4)] = scales[my_idx2];
	else if(idx_1d < ld_div_4*4) b2[idx_1d%(ld_div_4)] = biases[my_idx2];

	u_m.vec = m2; 
	u_v.vec = v2;
	u_s.vec = s2;
	u_b.vec = b2;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(i = 0; i<LOCAL_DEPTH; i++){
		ans[i] = (ans[i] - u_m.arr[i]) / (sqrt(u_v.arr[i]) + .000001f);
		ans[i] = ans[i]*u_s.arr[i] + u_b.arr[i];

		if(ans[i] < 0) ans[i] *= 0.1f;
		imageC[inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x] = ans[i];
		//vstore_half(ans[i], inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x, imageC);
	}
	
}
__kernel void Conv3_vec(__global float2* imageA, __global float2* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, 
				   __global const float* mean, __global const float* variance,
				   __global const float* scales, __global const float* biases)
{
	__local float2 localImage[85];
	float2 keep;
	__local float2 localFilter[36];
	//__local float localFilter[8][9];
	// 0 : z, 1: y, 2 : x
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int i = 0;
	int j = 0;
	int k = 0;
	float myImage[3][3];
	float ans[LOCAL_DEPTH] = {0.0}; 
	// check if initialization here is possible
	float mine;
	int my_idx, my_idx2;
	__local float m2[LOCAL_DEPTH];
	__local float v2[LOCAL_DEPTH];
	__local float s2[LOCAL_DEPTH];
	__local float b2[LOCAL_DEPTH];
	int idx_1d = global_y * 13 + global_x;
	int chunk_num = idx_1d / 9;
	int inner_i;
	int i_mod_2, my_i, my_j, my_i_from, my_j_from;
	lptr2 u_img, u_filt;

	for( i = 0; i< inputChannel; i++){
		i_mod_2 = i%2;
		 
		if(idx_1d < 85){
			my_idx = (int)(i/2)*169 + 85*i_mod_2 + idx_1d;
			
			if(idx_1d == 84){
				if(i_mod_2 == 0){
					localImage[idx_1d] = imageA[my_idx];
					keep = localImage[idx_1d]; 
				}
				else{
					localImage[0] = keep;
				}
			}
			else{
				localImage[idx_1d + i_mod_2] = imageA[my_idx];
			}
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		u_img.vec = localImage;

		///////////////////////////// copy image to private ////////////////////////////////////////////////////////
		
		my_i_from = global_y - 1;
		my_j_from = global_x - 1;
		for(j = 0; j<3; j++){
			my_i = my_i_from + j;
			my_j = global_x - 1;
			for(k = 0; k<3; k++){
				my_j = my_j_from + k;
				if(my_i < 0 || my_j < 0 || my_i > 12 || my_j > 12) myImage[j][k] = 0.0f;
				else myImage[j][k] = u_img.arr[13* my_i + my_j + i_mod_2];
			}
		}

		///////////////////////////// copy filter /////////////////////////////////////////
			
			////////////////////// scalar version ///////////////////////
			/*
			for(j = 1; j < LOCAL_DEPTH + 1; j++){
				if(idx_1d >= (j-1)*9 && idx_1d < j*9)
					localFilter[chunk_num][idx_1d%9] = vload_half((LOCAL_DEPTH*global_z+chunk_num)*9*inputChannel + i*9 + idx_1d%9, imageB);
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			*/
			////////////////////// vector version ///////////////////////
			
			if(idx_1d < 36){
				my_idx = (9*inputChannel*8*global_z + 72*i)/2 + idx_1d;
				localFilter[idx_1d] = imageB[my_idx];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			u_filt.vec = localFilter;
			

		
		///////////////////////////////////// convolution starts /////////////////////////
		for(inner_i = 0; inner_i < LOCAL_DEPTH; inner_i++){
			for (j = 0 ; j<3; j++){
				for(k = 0; k <3 ; k++){
					ans[inner_i] += (u_filt.arr[9 * inner_i + 3*j+k] * myImage[j][k]);
					//ans[inner_i] += (localFilter[inner_i][3*j+k] * myImage[j][k]);
				}
			}	
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	/*
	if(global_x == 0 && global_y == 0 && global_z == 0){ 
		
		//for(j = 0; j < 10; j++) printf("%f ", vload_half(j, imageA));
		//printf("\n");
		//float2 f1 = vload_half2(1, imageA); float2 f2 = vload_half2(2, imageA);
		//printf("f1 : %f %f, f2 : %f %f\n", f1.x, f1.y, f2.x, f2.y);
		//printf("look! : %f %f %f %f, imod2 = %d\n", u_img.arr[0],u_img.arr[1],u_img.arr[2],u_img.arr[3], i_mod_2);
		printf("%f %f %f\n%f %f %f\n%f %f %f\n", myImage[0][0], myImage[0][1], myImage[0][2], myImage[1][0], myImage[1][1], myImage[1][2], myImage[2][0], myImage[2][1], myImage[2][2]);
	}
	*/
	//if(idx_1d == 84) printf("%f %f\n", keep.x, keep.y);

	my_idx2 = LOCAL_DEPTH*global_z + idx_1d%LOCAL_DEPTH;
	if(idx_1d < LOCAL_DEPTH) m2[idx_1d] = mean[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*2) v2[idx_1d%LOCAL_DEPTH] = variance[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*3) s2[idx_1d%LOCAL_DEPTH] = scales[my_idx2];
	else if(idx_1d < LOCAL_DEPTH*4) b2[idx_1d%LOCAL_DEPTH] = biases[my_idx2];

	barrier(CLK_LOCAL_MEM_FENCE);

	for(i = 0; i<LOCAL_DEPTH; i++){
		ans[i] = (ans[i] - m2[i]) / (sqrt(v2[i]) + .000001f);
		ans[i] = ans[i]*s2[i] + b2[i];

		if(ans[i] < 0) ans[i] *= 0.1f;
		imageC[inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x] = ans[i];
		//vstore_half(ans[i], inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x, imageC);
		
	}
	
}

__kernel void Conv3(__global float* imageA, __global float* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, 
				   __global const float* mean, __global const float* variance,
				   __global const float* scales, __global const float* biases)
{
	__local float localImage[15][15];
	__local float localFilter[LOCAL_DEPTH][9];
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
	float myImage[3][3];
	float ans[LOCAL_DEPTH] = {0.0}; 
	// check if initialization chere is possible
	float mine;
	int my_idx, my_idx2;
	__local int pad_below, pad_above, pad_right, pad_left;
	__local float m2[LOCAL_DEPTH];
	__local float v2[LOCAL_DEPTH];
	__local float s2[LOCAL_DEPTH];
	__local float b2[LOCAL_DEPTH];
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
		
		///////////////////////////// copy image to private ////////////////////////////////////////////////////////
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
		ans[i] = (ans[i] - m2[i]) / (sqrt(v2[i]) + .000001f);
		ans[i] = ans[i]*s2[i] + b2[i];

		if(ans[i] < 0) ans[i] *= 0.1f;
		imageC[inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x] = ans[i];
		//vstore_half(ans[i], inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x, imageC);
	}
}

__kernel void in_Conv3(__global unsigned char* imageA, __global float* imageB, __global float* imageC, 
				   const int inputWidth, const int inputHeight, const int inputChannel, 
				   const int filter_num, 
				   __global const float* mean, __global const float* variance,
				   __global const float* scales, __global const float* biases)
{
	__local unsigned char localImage[15][15];
	__local float localFilter[LOCAL_DEPTH][9];
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
	unsigned char myImage[3][3];
	float ans[LOCAL_DEPTH] = {0.0}; 
	// check if initialization chere is possible
	unsigned char mine;
	int my_idx, my_idx2;
	__local int pad_below, pad_above, pad_right, pad_left;
	__local float m2[LOCAL_DEPTH];
	__local float v2[LOCAL_DEPTH];
	__local float s2[LOCAL_DEPTH];
	__local float b2[LOCAL_DEPTH];
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
		for(k = 0; k<15; k++) localImage[0][k] = 0;
		pad_above = 1;
	}
	
	else if(global_y == inputHeight -1 && local_x == 0 && local_y == LOCAL_HEIGHT - 1){
		for(k = 0; k<15; k++) localImage[14][k] = 0;
	//	if(global_x < 20 && global_y < 20 && global_z == 0)
	//	printf("padding below, global : %d %d %d, local : %d %d\n", global_z, global_y, global_x, local_x, local_y);
		pad_below = 1;
	}
	
	if(global_x == 0 && local_x == 0 && local_y == 0 ){
		for(k = 0; k<15; k++) localImage[k][0] = 0;
		pad_left = 1;
	}
	else if(global_x == inputWidth -1 && local_x == LOCAL_WIDTH -1 && local_y == LOCAL_HEIGHT - 1){
		for(k = 0; k<15; k++) localImage[k][14] = 0;
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
		
		///////////////////////////// copy image to private ////////////////////////////////////////////////////////
		for(j = 0; j<3; j++){
			for(k = 0; k<3; k++){
				myImage[j][k] = localImage[local_y+j][local_x+k];
			}
		}
		///////////////////////////////////// convolution starts /////////////////////////
		for(inner_i = 0; inner_i < LOCAL_DEPTH; inner_i++){	
			for (j = 0 ; j < 3; j++){
				for(k = 0; k < 3 ; k++){
					ans[inner_i] += (localFilter[inner_i][3*j+k]*myImage[j][k]*0.003921569f);
					// ans[inner_i] += (localFilter[inner_i][3*j+k]*myImage[j][k]/255.f);
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
		ans[i] = (ans[i] - m2[i]) / (sqrt(v2[i]) + .000001f);
		ans[i] = ans[i]*s2[i] + b2[i];

		if(ans[i] < 0) ans[i] *= 0.1f;
		imageC[inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x] = ans[i];
		//vstore_half(ans[i], inputWidth*inputHeight*(LOCAL_DEPTH*global_z + i) + inputWidth*global_y + global_x, imageC);
	}	
}

__kernel void Pool2(__global float* imageC, __global float* imageA)
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
	//vstore_half(v, my_idx, imageA);
}


__kernel void Pool(__global float* imageC, __global float* imageA)
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
			//vstore_half(ans, idx, imageA);
			
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
		//imageA[my_idx] = imageC[my_idx];
		vstore_half(vload_half(my_idx, imageC), my_idx, imageA);
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

__kernel void test1(__global float* input, const int channel_num){

	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);

	int i, j;
	__local float image[13][13];

	for(i=0; i< channel_num; i++){
		image[global_y][global_x] = input[169*i + 13*global_y + global_x];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void test2(__global float* input, const int channel_num){

	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);

	int i, j;
	__local float image[13][13];
	
	for(i=0; i< channel_num; i++){
		j = (global_z+i)%channel_num;
		image[global_y][global_x] = input[169* + 13*global_y + global_x];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void test3(__global float* weight, const int channel_num){

	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	int idx_1d = 13*global_y + global_x;
	int i, j;
	__local float localFilter[8][9];

	for(i=0; i< channel_num; i++){
		if(idx_1d < 72)
			localFilter[idx_1d/9][idx_1d%9] = weight[(8*global_z + (int)(idx_1d/9)) * 9 * channel_num + 9*channel_num + idx_1d%9];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


__kernel void test4(__global float* weight, const int channel_num, const int out_channel){

	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	int idx_1d = 13*global_y + global_x;
	int i, j;
	__local float localFilter[8][9];

	for(i=0; i< channel_num; i++){
		if(idx_1d < 72)
			//localFilter[idx_1d/9][idx_1d%9] = weight[(8*global_z)*9*channel_num + 72*i + idx_1d];
			localFilter[idx_1d/9][idx_1d%9] = weight[9*out_channel*i + global_z*72 + idx_1d];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void test5(__global ulong* imageA, __global ulong* imageB, const int inputChannel)
{
	__local ulong localImage[43];
	__local ulong localFilter[18];
	//__local float localFilter[8][9];
	// 0 : z, 1: y, 2 : x
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int i = 0;
	int j = 0;
	int k = 0;
	
	// check if initialization here is possible
	float mine;
	int my_idx, my_idx2;
	
	int idx_1d = global_y * 13 + global_x;
	
	int inner_i;
	int i_mod_4, my_i, my_j, my_i_from, my_j_from;
	

	for( i = 0; i< inputChannel; i++){
		i_mod_4 = i%4;
		 
		if(idx_1d < 43){
			my_idx = (int)(i/4)*169 + 42*i_mod_4 + idx_1d;
			
			localImage[idx_1d] = imageA[my_idx];
			
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	

		///////////////////////////// copy filter /////////////////////////////////////////
			
			////////////////////// scalar version ///////////////////////
			/*
			if(idx_1d < 9*LOCAL_DEPTH){
			localFilter[chunk_num][idx_1d%9] = vload_half((LOCAL_DEPTH*global_z+chunk_num)*9*inputChannel + i*9 + idx_1d%9, imageB);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			*/
			////////////////////// vector version ///////////////////////
			
			if(idx_1d < 18){
				my_idx = (9*inputChannel*8*global_z + 72*i)/4 + idx_1d;
				localFilter[idx_1d] = imageB[my_idx];

			}
			barrier(CLK_LOCAL_MEM_FENCE);

	}

}

__kernel void test6(__global half* imageA, __global half* imageB, const int inputChannel)
{
	__local float4 localImage[43];
	__local float4 localFilter[18];
	//__local float localFilter[8][9];
	// 0 : z, 1: y, 2 : x
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int i = 0;
	int j = 0;
	int k = 0;
	
	// check if initialization here is possible
	float mine;
	int my_idx, my_idx2;
	
	int idx_1d = global_y * 13 + global_x;
	
	int i_mod_4, my_i, my_j, my_i_from, my_j_from;
	

	for( i = 0; i< inputChannel; i++){
		i_mod_4 = i%4;
		 
		if(idx_1d < 43){
			my_idx = (int)(i/4)*169 + 42*i_mod_4 + idx_1d;
			
			localImage[idx_1d] = vload_half4(my_idx, imageA);
			
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	

		///////////////////////////// copy filter /////////////////////////////////////////
			
			////////////////////// scalar version ///////////////////////
			/*
			if(idx_1d < 9*LOCAL_DEPTH){
			localFilter[chunk_num][idx_1d%9] = vload_half((LOCAL_DEPTH*global_z+chunk_num)*9*inputChannel + i*9 + idx_1d%9, imageB);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			*/
			////////////////////// vector version ///////////////////////
			
			if(idx_1d < 18){
				my_idx = (9*inputChannel*8*global_z + 72*i)/4 + idx_1d;
				localFilter[idx_1d] = vload_half4(my_idx, imageB);

			}
			barrier(CLK_LOCAL_MEM_FENCE);

	}

}