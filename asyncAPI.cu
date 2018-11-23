#include <stdio.h>

__global__ void matrixs_1D_multiplication(int *matrix_a_dev,int *matrix_b_dev,int *matrix_c_dev,int row,int col)//记住这里的row和col直接对应global里面的数值，不能有误
{
    int j = threadIdx.x+blockIdx.x * blockDim.x;
    int i = threadIdx.y+blockIdx.y * blockDim.y;

    if(i< row &&j < row)
    {
        for(int k = 0; k < col; k++)
        {
            matrix_c_dev[row *i + j] += matrix_a_dev[i* col + k] * matrix_b_dev[row*k + j];
        }
    }
}

int main()
{
    int row = 4;
    int col = 5;
    int *matrix_a_host;
    int *matrix_b_host;
    int *matrix_c_host;

    matrix_a_host = (int *)malloc(row*col*sizeof(int));
    matrix_b_host = (int *)malloc(row*col*sizeof(int));
    matrix_c_host = (int *)malloc(row*row*sizeof(int));

    for(int i = 0; i<row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            matrix_a_host[i*col +j] = i+j;
        }
    }

    printf("\n-------------Matrix a-----------------\n");
    for(int i = 0; i < row*col; i++)
    {
        printf("%d ",*(matrix_a_host + i));
        if(i%col==col-1) printf("\n");//每输出3个换行。
    }


    for(int i = 0; i<col; i++)
    {
        for(int j = 0; j < row; j++)
        {
            matrix_b_host[i*row +j] = i+j;
        }
    }



    // ------------------GPU--------------------------
    int *matrix_a_dev;
    int *matrix_b_dev;
    int *matrix_c_dev;

    cudaMalloc((void**) &matrix_a_dev, row*col*sizeof(int));
    cudaMalloc((void**) &matrix_b_dev, row*col*sizeof(int));
    cudaMalloc((void**) &matrix_c_dev, row*row*sizeof(int));

    cudaMemcpy(matrix_a_dev, matrix_a_host, row*col*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_b_dev, matrix_b_host, row*col*sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 2, 1);
    dim3 dimBlock(4, 5, 1);//记住这里的row和col直接对应global里面的数值，不能有误
    matrixs_1D_multiplication<<<dimGrid, dimBlock>>>(matrix_a_dev, matrix_b_dev, matrix_c_dev, row,col);

    cudaMemcpy(matrix_c_host, matrix_c_dev, row*row*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n-------------Matrix c-----------------\n");
    for(int i = 0; i < row*row; i++)
    {
        printf("%d ",*(matrix_c_host + i));
        if(i%row==row-1) printf("\n");//每输出4个换行。
    }

    free(matrix_a_host);
    free(matrix_b_host);
    free(matrix_c_host);
    cudaFree(matrix_a_dev);
    cudaFree(matrix_b_dev);
    cudaFree(matrix_c_dev);

    return 1;
}
