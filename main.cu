//Jan Morez, Master in Physics, University of Antwerp, 
//January, 2014

//Homemade functions & STL headers
#include "Header.h"
#include <cmath>
//GPU related includes
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>

//Errorchecking wrapper for CUDA API calls.
#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
		  exit(code);
	  }
   }
}

//Same for CUBLAS
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__);}
inline void cublasAssert(cublasStatus_t status,const char* file,int line){
	if (_cudaGetErrorEnum(status) == "CUBLAS_STATUS_SUCCESS"){ 
		printf("CUBLAS Error: %s in file %s on line %i \n",_cudaGetErrorEnum(status),file,line);
	}
}

/*
========================================================================================
Short explanation of this whole thing's purpose: investigate the computational 
complexity as a function of inputs, hidden units and outputs for a Multi-Layer-Perceptron with one hidden unit layer. 
Short outline of the main function:
	1)Read training set data from a binary file.
	2)Arrange everything so the GPU can start crunching matrices (device data, random weight matrices,...).
	3)Uses gradient-descent to train the network. 
	4)Return the trained network matrices w_ji and w_kj after a suitable error is encountered.
	5)Benchmark steps 3 & 4, ignoring overhead from memory transfers, etc. 
========================================================================================	
GENERAL APPLICATION OUTLINE
==	Initialization of variables
|	Constants, arrays, etc.
|	
|
==	Learning loop
|
||==Calculation loop for each pattern: calculate the output, compare it with the training pattern and find the backpropagation matrix
||  
||
||==
|
==	Destroy variables, etc. 
|
|	
|
==

*/

//=== USEFUL KERNELS===

//Parallel(element-wise) activation function
__global__ void activation(float* d_array,int size,float* d_result){
	int i=(threadIdx.x)+(blockIdx.x)*(blockDim.x);
	if(i<size){
	d_result[i]=atan(d_array[i]);
	}
}
//Element-wise derivative of the activation function
__global__ void activation_derivative(float* d_zj,int size,float* d_result){
	int i=(threadIdx.x)+(blockIdx.x)*(blockDim.x);
	if(i<size){
	d_result[i]=d_zj[i]*(1-d_zj[i]);
	}
}
//Elementwise array product
__global__ void ewarrayprod(float* d_A,float* d_B,int size, float* d_C){
	int i=(threadIdx.x)+(blockIdx.x)*(blockDim.x);
	if(i<size){
		d_C[i]=d_A[i]*d_B[i];
	}
}
//Set an entire d_target array to a d_value
__global__ void set_array(float* d_value,int size, float* d_target){
	int i=(threadIdx.x)+(blockIdx.x)*(blockDim.x);
	if(i<size){
		d_target[i]=d_value[0];
	}
}
//Elementwise array cumulative sum
__global__ void ewarraycumsum(float* d_A,int size, float* d_C){
	int i=(threadIdx.x)+(blockIdx.x)*(blockDim.x);
	if(i<size){
		d_C[i]+=d_A[i];
	}
}
//ERF vector!
__global__ void vectorDiffSq(const float* d_A,const float* d_B,int size,float* d_C){
	int i=(threadIdx.x)+(blockIdx.x)*(blockDim.x);
	if (i < size){
	d_C[i]=((float)0.5)*pow(d_A[i]-d_B[i],(float)2);
	}
}
	//Memory location naming convention: host variables will start with the affix h_, device variables with d_.
	//Example: h_X_v is a vector variable located in host memory called "X".

int main()
{ 
	//===================================PART 1: INITIALIZATION OF CONSTANTS===================================
	//HOST CONSTANTS
	//Perceptron-specific variables
	const size_t h_inputs=2;
	const size_t h_hidden=2048*16;
	const size_t h_outputs=1;
	const size_t h_patterns=4;

	//Training parameters
	const size_t max_iterations=10;
	float eta0=0.00001; //Learning parameter, can be immediately applied when doing the outer products. The power of CUBLAS!
	const float epsilon=0.01; //Error criterion, will be used in a while loop. 
	unsigned long seed=2; //Not really a parameter. It is the seed that generates the random w_ji and w_kj matrices
	bool dynamic_learning=true; //
	
	//Parameters for dynamic learning, see p.9 of the paper
	float lp_rate=0.0001;
	float lp_at_infinity=0.5;

	//Constant BLAS parameters
	cublasHandle_t handle;
	cublasErrchk(cublasCreate(&handle));

	const cublasOperation_t trans=CUBLAS_OP_T;
	const cublasOperation_t n_trans=CUBLAS_OP_N;
	const float alpha=1.;
	const float minus_alpha=-1.;
	const float beta=0;
	const int incx=1;
	const int incy=1;

	//Kernel parameters
	//Figure out grid layout for all the kernel calls (1D blocks for element-wise operations on a linear array).
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);
	const int MaxThreadsPerBlock=device.maxThreadsPerBlock;
	dim3 BlockDim;
	BlockDim.x=MaxThreadsPerBlock;

	//HOST VARIABLES
	//Initialize variables to calculate the total error (i.e. all patterns). The error will be calculated with
	//a kernel,therefore we will need to write a d_array to a h_array.
	//Both these variables need to be set to zero in each training iteration!!!
	float h_E_single_pattern[]={0.};
	float h_E_all_patterns0=0.;
	float h_E_all_patterns1=0.;
	float h_Errorchange=-1;
	//===================================END OF PART 1======================================================



	//===================================PART 2: TRANSFER TO DEVICE MEMORY===================================


	//DEVICE CONSTANTS
	//Device array with one element set to "1", is needed to add the bias input that is permanently activated.
	float h_activation=1;
	float* d_activation;
	cudaErrchk(cudaMalloc((void**)&d_activation,sizeof(float)));
	cudaErrchk(cudaMemcpy(d_activation,&h_activation,sizeof(float),cudaMemcpyHostToDevice));

	//Will be useful to set arrays to zero
	float h_zero=0;
	float* d_zero;
	cudaErrchk(cudaMalloc((void**)&d_zero,sizeof(float)));
	cudaErrchk(cudaMemcpy(d_zero,&h_zero,sizeof(float),cudaMemcpyHostToDevice));


	//TRAINING DATA
	/*patterns.bin is assumed to contain the "input pattern matrix" in colum-major binary form, 
	which is inputs*patterns*sizeof(float) bytes large. The original matrix in Matlab had <h_inputs> rows and <h_patterns> columns.*/	
	//float* h_inputpatterns_mPtr=read_bin_data("patterns.bin");
	
	//For now, we will manually initialize this matrix to make sure no external mistakes can affect any further calculations.
	const float h_inputpatterns_m[]={0.,0.,1.,1.,0.,1.,1.,0.};
	const float h_outputpatterns_m[]={0.,0.,1.,1.};
	
	//Try with an extra input
	//const float h_inputpatterns_m[]={0.,0.,0.,1.,1.,1,0.,1.,1.,0.,1.,0.,};
	//const float h_outputpatterns_m[]={0.,0.,1.,1.};
	
	//Try with an extra pattern
	//const float h_inputpatterns_m[]={0.,0.,1.,1.,0.,1.,1.,0.,0.5,0.5};
	//const float h_outputpatterns_m[]={0.,0.,1.,1.,0.7};

	//Allocate device memory.
	float* d_inputpatterns_m;
	cudaErrchk(cudaMalloc((void**)&d_inputpatterns_m,h_patterns*h_inputs*sizeof(float)));
	float* d_outputpatterns_m;
	cudaErrchk(cudaMalloc((void**)&d_outputpatterns_m,h_patterns*h_outputs*sizeof(float)));

	//Write data to device
	cublasErrchk(cublasSetMatrix(h_inputs,h_patterns,sizeof(float),h_inputpatterns_m,h_inputs,d_inputpatterns_m,h_inputs));
	cublasErrchk(cublasSetMatrix(h_outputs,h_patterns,sizeof(float),h_outputpatterns_m,h_outputs,d_outputpatterns_m,h_outputs));


	//DEVICE VARIABLES

	//Initialize input vector including bias
	float* d_x_v;
	cudaErrchk(cudaMalloc((void**)&d_x_v,(h_inputs+1)*sizeof(float)));

	//Initialize the hidden layer value variables
	float* d_aj_v;
	cudaErrchk(cudaMalloc((void**)&d_aj_v,(h_hidden)*sizeof(float)));

	//Initialize the hidden layer activation, without bias
	float * d_zj_v;
	cudaErrchk(cudaMalloc((void**) &d_zj_v,h_hidden*sizeof(float)));

	//Initialize the hidden layer value activation, with bias
	float* d_zj_bias_v;
	cudaMalloc((void**) &d_zj_bias_v,(h_hidden+1)*sizeof(float));

	//Initialize wji and wkj matrices and fill them with random numbers. Account for bias!!!
	float* d_wji_m;
	float* d_wkj_m;
	cudaErrchk(cudaMalloc((void**)&d_wji_m,(h_inputs+1)*h_hidden*sizeof(float)));
	cudaErrchk(cudaMalloc((void**)&d_wkj_m,(h_hidden+1)*h_outputs*sizeof(float)));



	//Fill these with random numbers using CURAND
	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,seed);
	curandGenerateUniform(gen, d_wji_m, (h_inputs+1)*h_hidden);
	curandGenerateUniform(gen, d_wkj_m, (h_hidden+1)*h_outputs);

	//printf("Showing d_wkj_m \n");
	//show_gpu_array(d_wkj_m,3);

	//Intermediate error vector, needs to be set to zero for each training iteration too!!!
	float* d_Ek;
	cudaErrchk(cudaMalloc((void**) &d_Ek,(h_outputs)*sizeof(float)));
	cudaErrchk(cudaMemcpy(d_Ek,d_zero,(h_outputs)*sizeof(float),cudaMemcpyDeviceToDevice));
	
	//Gradient descent variables
	float* d_deltak;
	cudaErrchk(cudaMalloc((void**)&d_deltak,h_outputs*sizeof(float)));

	float* d_deltaj1;
	cudaErrchk(cudaMalloc((void**) &d_deltaj1,(h_hidden+1)*sizeof(float)));

	float* d_deltaj2;
	cudaErrchk(cudaMalloc((void**) &d_deltaj2,(h_hidden+1)*sizeof(float)));

	float* d_deltaj;
	cudaErrchk(cudaMalloc((void**) &d_deltaj,(h_hidden+1)*sizeof(float)));

	//Initialize the matrix targets to which each corrective matrix of each pattern will be added
	float* d_deltawkj_all_patterns;
	cudaErrchk(cudaMalloc((void**) &d_deltawkj_all_patterns,(h_hidden+1)*h_outputs*sizeof(float)));
	float* d_deltawji_all_patterns;
	cudaErrchk(cudaMalloc((void**) &d_deltawji_all_patterns,(h_inputs)*(h_hidden+1)*sizeof(float)));

	//These are the correction matrices for a single pattern
	float* d_deltawkj_single_pattern;
	cudaErrchk(cudaMalloc((void**) &d_deltawkj_single_pattern,(h_hidden+1)*h_outputs*sizeof(float)));

	float* d_deltawji_single_pattern;
	cudaErrchk(cudaMalloc((void**) &d_deltawji_single_pattern,(h_inputs+1)*(h_hidden)*sizeof(float)));


	//Profiling code
	cudaEvent_t startG,stopG;
	cudaErrchk(cudaEventCreate(&startG));
	cudaErrchk(cudaEventCreate(&stopG));
	float timeG=0.;
	cudaEventRecord(startG,0);
	printf("Starting training loop and timer. \n");

	int k=0;
	//Start of the training loop
	float eta=eta0;
	int GridSize;
			if (h_hidden > 1024){
			GridSize=(int)ceil((float)(h_hidden+1)/MaxThreadsPerBlock);
		}
		else{
			GridSize=(int)ceil((float)(h_hidden)/MaxThreadsPerBlock);
		}
	do {
		k+=1; 
		if (dynamic_learning==true){
			eta=(lp_at_infinity-eta0)*tanh(lp_rate*(float)k/1000.)-eta0);
		}
		else{
			eta=-eta0;
		}

		//Note that GridSize will change depending on what you want to process with a kernel!!!
		//Since the amount of outputs will always be less than the amount of hidden units, it's 
		//safe to use the same GridSize for d_deltawkj_.. and d_deltaji_..
		
		//This set of conditions seems to fix... something? 
		if (h_hidden > 1024){
			GridSize=(int)ceil((float)(h_hidden+1)/MaxThreadsPerBlock);
		}
		else{
			GridSize=(int)ceil((float)(h_hidden)/MaxThreadsPerBlock);
		}
		//Set the correction matrices to zero
		set_array<<<GridSize,BlockDim>>>(d_zero,(h_hidden+1)*h_outputs,d_deltawkj_all_patterns);
		set_array<<<GridSize,BlockDim>>>(d_zero,(h_inputs+1)*h_hidden,d_deltawji_all_patterns);
		set_array<<<GridSize,BlockDim>>>(d_zero,(h_hidden+1)*(h_outputs),d_deltawkj_single_pattern);
		set_array<<<GridSize,BlockDim>>>(d_zero,(h_inputs+1)*(h_hidden),d_deltawji_single_pattern);
		//Reset the error function value
		//Calculate the change in error
		h_E_all_patterns1=h_E_all_patterns0;
		h_E_all_patterns0=0.;
	//Calculate y_k for a single pattern.
	for (int n=0;n<h_patterns;n++){
		//printf("Now calculating the output for pattern %i \n",n);

		//Bias is a permanently activated input. To avoid any memory transfer-overhead, we use cudaMemcpyDeviceToDevice
		//to "paste" a 1 in front of the actual input array. 
		//Set the input vector. 

		cudaErrchk(cudaMemcpy(d_x_v,d_activation,sizeof(float),cudaMemcpyDeviceToDevice));
		cudaErrchk(cudaMemcpy(d_x_v+1,d_inputpatterns_m+n*h_inputs,h_inputs*sizeof(float),cudaMemcpyDeviceToDevice));	

		//Calculate a_j=w_ji * x_i, or generally y=Ax
		cublasErrchk(cublasSgemv(	handle,		
									n_trans,		
									h_hidden,	//Amount of rows of op(A)
									h_inputs+1,	//Amount of columns of A, including bias
									&alpha,		
									d_wji_m,	//A
									h_hidden,	//Leading dimension of A, in CM-storage the memory distance between 2 columns, i.e. the number of rows
									d_x_v,		//x
									incx,
									&beta,
									d_aj_v,		//y
									incy));

		//Calculate the activation z_j=g(a_j) with a kernel
		//GridSize is still correct because the size of d_zj_v is still h_hidden
		activation<<<GridSize,BlockDim>>>(d_aj_v,h_hidden,d_zj_v);
		cudaDeviceSynchronize();

		//Account for bias by sticking a 1 in front of z_j!
		cudaErrchk(cudaMemcpy(d_zj_bias_v,d_activation,sizeof(float),cudaMemcpyDeviceToDevice));
		cudaErrchk(cudaMemcpy(d_zj_bias_v+1,d_zj_v,(h_hidden)*sizeof(float),cudaMemcpyDeviceToDevice));

		//printf("Showing d_zj_bias_v \n");
		//show_gpu_array(d_zj_bias_v,3);

		//Calculate y_k=w_kj * zj

		float* d_yk_v;
		cudaErrchk(cudaMalloc((void**)&d_yk_v,(h_outputs)*sizeof(float)));
		//y=Ax with y,x vectors and A a matrix
		cublasErrchk(cublasSgemv(	handle,		
									n_trans,		
									h_outputs,	//Amount of rows of A
									h_hidden+1,	//Amount of columns of A
									&alpha,		
									d_wkj_m,	//Matrix A
									h_outputs,	//lda
									d_zj_bias_v,//x
									incx,		
									&beta,		
									d_yk_v,		//y
									incy));		
		
		//printf("Showing y_k: \n");
		//show_gpu_array(d_yk_v,1);
		

		//Calculate error, returns a vector with components d_k=0.5*(y_k-t_k)^2,these get summed with cublasDasum()
		GridSize=(int)ceil((float)h_outputs/MaxThreadsPerBlock);
		vectorDiffSq<<<GridSize,BlockDim>>>(d_outputpatterns_m+n*h_outputs,d_yk_v,h_outputs,d_Ek);

		//d_Ek is the error for a single pattern for a single output unit, we need to
		//sum this vector to find the total error for a single pattern. 
		cublasErrchk(cublasSasum(handle,h_outputs,d_Ek,incx,h_E_single_pattern));


		//Write the total error for all patterns to a variable outside of this loop. 
		//h_E_single_pattern is somewhat superfluous, but allows us to check if the error gets calculated correctly.
		h_E_all_patterns0+=h_E_single_pattern[0];

		//=========================Gradient descent part====================


		//calculate delta_k=y_k-t_k, or just the difference of the vectors y and t
		cublasErrchk(cublasSgeam(	handle,
									n_trans, 
									n_trans,
									h_outputs,		//int m, rows of A
									1,				//int n, columns of B and C
									&alpha,			//const float* alpha,
									d_yk_v,			//const float* A
									h_outputs,		//int lda, i.e. the number of rows in A
									&minus_alpha,	//const float * beta
									d_outputpatterns_m+n*h_outputs,  //const float* pointer to B
									h_outputs,		//int ldB, 
									d_deltak,		//float* C
									h_outputs));	//int ldC

		//printf("Showing delta_k \n");
		//show_gpu_array(d_deltak,1);

		//Calculate delta_j=z_j(1-z_j) SUM_k[w_kj*d_k]=d_deltaj1*d_deltaj2

		//First calculate z_j*(1-z_j)=d_deltaj1

		if (h_hidden > 1024){
			GridSize=(int)ceil((float)(h_hidden+1)/MaxThreadsPerBlock);
		}
		else{
			GridSize=(int)ceil((float)(h_hidden)/MaxThreadsPerBlock);
		}
		activation_derivative<<<GridSize,BlockDim>>>(d_zj_bias_v,h_hidden+1,d_deltaj1);
		cudaDeviceSynchronize();

		//Calculate w_kj'*dk=d_deltaj2
		
		cublasErrchk(cublasSgemv(	handle,			
									trans,			//note that w_kj*d_k=w_jk'*d_k, so we do need a transpose in this case
									h_outputs,		//rows of op(A)
									h_hidden+1,		//columns of A
									&alpha,			//alpha=1
									d_wkj_m,		//A
									h_outputs,		//ldA
									d_deltak,
									incx,
									&beta,
									d_deltaj2,
									incy));

		//Calculate d_j=d_deltaj1*d_deltaj2

		if (h_hidden > 1024){
			GridSize=(int)ceil((float)(h_hidden+1)/MaxThreadsPerBlock);
		}
		else{
			GridSize=(int)ceil((float)(h_hidden)/MaxThreadsPerBlock);
		}
		ewarrayprod<<<GridSize,BlockDim>>>(d_deltaj1,d_deltaj2,h_hidden+1,d_deltaj);
		cudaDeviceSynchronize();

		//printf("Showing d_deltaj \n");
		//show_gpu_array(d_deltaj,3);

		//Now we need an outer product, d_wkj=eta X delta_k*z_j'
		cublasErrchk(cublasSgemm(	handle,
									n_trans,		//op(A)
									trans,			//op(B)
									h_outputs,		//rows of op(A)
									h_hidden+1,		//columns of op(B)
									1,				//columns of op(A)
									&eta,			//immediately apply learning parameter
									d_deltak,		//A
									h_outputs,		//ldA
									d_zj_bias_v,			//B 
									h_hidden+1,				//ldB
									&beta,			//beta
									d_deltawkj_single_pattern,			//C
									h_outputs));	//ldC
		
		//printf("d_deltawkj_single_pattern \n");
		//show_gpu_array(d_deltawkj_single_pattern,(h_hidden+1)*h_outputs);
		
		//The second outer product d_deltawji=eta*d_deltaj X d_x_i

		cublasErrchk(cublasSgemm(	handle,
									n_trans,		//op(A)
									trans,			//op(B)
									h_hidden,		//rows of op(A)
									h_inputs+1,		//columns of op(B)
									1,				//columns of op(A)
									&eta,			//immediately apply the learning parameter
									d_deltaj+1,		//A
									h_hidden,		//ldA
									d_x_v,			//B 
									h_inputs+1,		//ldB
									&beta,			//beta
									d_deltawji_single_pattern,			//C
									h_hidden));		//ldC
		
		//printf("d_deltawji_single_pattern \n");
		//show_gpu_array(d_deltawji_single_pattern,(h_inputs+1)*h_hidden);
		
		//Now we need to apply batch learning, i.e. sum these matrices over each pattern
		ewarraycumsum<<<GridSize,BlockDim>>>(d_deltawkj_single_pattern,(h_hidden+1)*(h_outputs)*sizeof(float),d_deltawkj_all_patterns);
		ewarraycumsum<<<GridSize,BlockDim>>>(d_deltawji_single_pattern,(h_inputs+1)*(h_hidden)*sizeof(float),d_deltawji_all_patterns);
																
	}
	//Evaluate the change in error
	if(k==1){
		h_E_all_patterns1=h_E_all_patterns0+100.;
	}
	h_Errorchange=h_E_all_patterns0-h_E_all_patterns1;
	//printf("Correction matrix Ekj\n");
	//show_gpu_array(d_deltawkj_all_patterns,(h_hidden+1)*(h_outputs));

	//Update w_ji and w_kj with the correction matrices
	ewarraycumsum<<<GridSize,BlockDim>>>(d_deltawkj_all_patterns,(h_hidden+1)*(h_outputs),d_wkj_m);
	ewarraycumsum<<<GridSize,BlockDim>>>(d_deltawji_all_patterns,(h_inputs+1)*(h_hidden),d_wji_m);
	cudaDeviceSynchronize();
	printf("Epoch No. %i, total error is: %f \n",k,h_E_all_patterns0);
	} while(h_E_all_patterns0 >= epsilon && h_Errorchange <=0.);
	
	cudaEventRecord(stopG);
	cudaErrchk(cudaEventSynchronize(stopG));
	cudaErrchk(cudaEventElapsedTime(&timeG,startG,stopG));
	printf("MLP Training took: %f s and %i iterations \n",timeG/1000,k);
	if (h_Errorchange > 0){
		printf("Training stopped because the total error suddenly became bigger. \n");
	}
	


	//There's some (a lot) others out there, can't be bothered to go looking for them #good#programming#practices
	cudaErrchk(cudaFree(d_activation));
	cudaErrchk(cudaFree(d_inputpatterns_m));
	cudaErrchk(cudaFree(d_wji_m));
	cudaErrchk(cudaFree(d_wkj_m));

	cublasErrchk(cublasDestroy(handle));
	printf("\n Press any key to close this window... \n");
	getchar();
    return 0;

}

