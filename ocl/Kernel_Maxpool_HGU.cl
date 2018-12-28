/**
  * @file       Kernel_Maxpool_HGU.cl
  * @author     Yechan, Jinju, Daewoo
  * @date       2018-12-03
  * @email      21400067@handong.edu
  * @brief      OpenCL kernel for                   		\n
                maxpooling of 'float' version	    		\n
                function 'Pool2' for maxpooling layer11		\n
                function 'Pool' for other maxpooling layers	\n
**/




// function Pool parameters
// 	imageC : image
// 	imageA : output
/********************maxpooling for layer 1,3,5,7,9****************************/
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

	/********************copy image global->local****************************/
	my_idx = global_z*g1*g2+global_y*g1+global_x;
	localImage[local_y][local_x] = imageC[my_idx];
	barrier(CLK_LOCAL_MEM_FENCE);
	

	/********************maxpooling****************************/
	if(local_x %2 == 0 && local_y %2 == 0){
		ans = localImage[local_y][local_x];
		for(i = 0; i <2; i++){
			for(j = 0; j <2; j++){
				if(ans < localImage[local_y + i][local_x + j]){
					ans = localImage[local_y + i][local_x + j];
				}
			}
		}

		/********************copy output image local->global****************************/
		idx = global_z*(g1/2)*(g2/2) + global_y/2* g1/2 + global_x/2;
		imageA[idx] = ans;
	}
}


// function Pool2 parameters
// 	imageC : image
// 	imageA : output
/********************maxpooling for layer 11 only****************************/
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

	/********************copy image global->local****************************/
	int my_idx = (13*13*global_z) + (13*global_y) + global_x;
	localImage[local_y][local_x] = imageC[my_idx];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	v = localImage[local_y][local_x];
	
	/********************maxpooling****************************/
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
	
	/********************copy output image local->global****************************/
	imageA[my_idx] = v;
}




