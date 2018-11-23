#include<stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

/**
 * 需求：需要把若干个一维数组传给核函数
 * 实现方法：在gpu生成一个一维的指针数组，每个元素指向一个普通一维数组。
 * 把该指针数组的地址传递给核函数。
 * 其实该指针数组充当二维数组的角色。
 */

__global__ void
testKernel(float ** pointerArray)
{
	printf("(%d,%d):%f\n",threadIdx.y,threadIdx.x,pointerArray[threadIdx.y][threadIdx.x]);
}


int main()
{

	int rows = 2;
	int cols = 2;


	float** host_2d = new float*[rows];
	float** dev_2d;
	for (int k = 0; k < rows; k++)
	{
		float* host_1d = new float[cols];
		//---------下面两个是测试数据，看看核函数printf输出的是否与这个一致------------------------------
		host_1d[0] = 99;
		host_1d[1] = 100;

		float* dev_1d ;
		cudaMalloc((void**)&dev_1d,sizeof(float)*cols);//该指针指向的是一个float数组
		cudaMemcpy(dev_1d,host_1d,sizeof(float)*cols,cudaMemcpyHostToDevice);

		host_2d[k] = dev_1d;
	}
	cudaMalloc((void**)&dev_2d,sizeof(float*)*rows);
	cudaMemcpy(dev_2d,host_2d,sizeof(float*)*rows,cudaMemcpyHostToDevice);

	dim3 threads(rows,cols);
	testKernel<<<1,threads>>>(dev_2d);

	cudaDeviceSynchronize();

	//记得释放内存啊，养成好习惯。这里省去这部分代码




return 0;
}
