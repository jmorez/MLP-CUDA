#include "Header.h"
#include <cuda_runtime.h>

//Some useful helper functions to import/export data
float* read_bin_data (const char *inputfile){
	printf("%==================== \n read_data: \n");

	//Associate inputfile with a stream via ifptr, open file as binary data.
	FILE *ifPtr;
	ifPtr=fopen(inputfile,"rb");

	//Report success or display error message. 
	if(ifPtr==NULL){
		perror("	Error reading inputfile ");
		return NULL;
	}
	else{
		printf("	Inputfile '%s' opened. \n",inputfile);

		//Determine file size
		fseek(ifPtr,0,SEEK_END);
		int ifSize=ftell(ifPtr);
		rewind(ifPtr);
		printf("	Filesize is %i bytes. \n",ifSize);

		//Initialize data pointer and allocate a sufficient amount of memory
		float* DataPtr= (float*) malloc(ifSize);

		if (DataPtr==NULL){
			perror("	Problem allocating memory.");
			printf("\n File will not be read. \n \n");
		}

		//If memory allocation went well, continue to read file.
		else{
			printf("	%i bytes of memory succesfully allocated at %p \n",ifSize,DataPtr);
			
			//Read binary data into a memoryblock of size ifSize, pointed to by Data

			//Mogelijk probleem: op andere systemen variëert sizeof(float) maar de data zal altijd 4 bytes/element zijn.
			int Length=ifSize/(sizeof(float));
			int Elements_Read=fread(DataPtr,4,Length,ifPtr);
			printf("	%i elements out of %i read. \n \n",Elements_Read,Length);
		}
		return DataPtr;
		fclose(ifPtr);
		free(ifPtr);
	}
}

void export_data(const char* inputfile, float* data, int& elements_amount){
	if(data!=NULL){
		FILE* OfPtr;
		OfPtr=fopen(inputfile,"wb+");
		if(OfPtr!=NULL){
			int elements_written=fwrite(data,sizeof(float),elements_amount,OfPtr);
			printf("export_data: %i elements successfully written. \n", elements_written);
			fclose(OfPtr);
		}
	}
	else
		printf("export_data: input pointer is NULL. \n");
}

void show_gpu_array(double* d_array,size_t size){
	//Note: if the argument size is wrong (i.e. does not correspond with d_array), all output will be wrong! Always get the size right!
	double* h_array=(double*) malloc(size*sizeof(double));
	cudaMemcpy(h_array, d_array, size*sizeof(double),cudaMemcpyDeviceToHost);
	for(size_t i = 0; i < size; i++) {
		printf("%f \n", h_array[i]);
	}
	printf("\n");
}


