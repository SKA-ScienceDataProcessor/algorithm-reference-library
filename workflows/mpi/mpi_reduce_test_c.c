#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <stddef.h>

#define N 4

typedef struct Images{
	int data[N];
	int weight[N];
}Image;


void printImage(Image myImage){
	printf("\nImage:\n\t");
	int i;
	for(i=0;i<N;i++){
		printf(" (%d, %d) ",myImage.data[i],myImage.weight[i]);
	}
}


// MPI_User_function
void mysum( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
	Image *image1=invec;
	Image *image2=inoutvec;
	int i,j;
	printf("In mysum\n");
	printImage(*image1);
	printImage(*image2);
	
	for(i=0;i<*len;i++){
		for(j=0;j<N;j++){
			int scale = image1->weight[j];
			printf("scale %d: %d\n",j,scale);
			image2->data[j] = image2->data[j]* image2->weight[j] + scale * image1->data[j];
			image2->weight[j] = 1;
		}
	}
	printf("After mysum\n");
	printImage(*image2);
}


int main(int argc, char** argv) {


	MPI_Init(NULL, NULL);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Create Image in each process
	Image myImage;
	int i;
	for(i=0;i<N;i++){
		myImage.data[i]=rank;
		myImage.weight[i]=i;
	}

	// Print the Image on each process
	//printf("\n%d:Image:\n\t", rank);
	//printImage(myImage);
	
	// Reduce the image into the global image 
	Image globalImage;
	for(i=0;i<N;i++){
		globalImage.data[i]=0;
		globalImage.weight[i]=0;
	}

	// Create the MPI_Datatype to Reduce Images
	MPI_Datatype image_dt;
	int structlen = 2;
	int blocklenghts[structlen]; MPI_Datatype types[structlen];
	MPI_Aint displacements[structlen];
	blocklenghts[0]=N; types[0]= MPI_INT;
	blocklenghts[1]=N; types[1]= MPI_INT;
	displacements[0]=offsetof(Image,data);
	displacements[1]=offsetof(Image,weight);

	MPI_Type_create_struct(structlen,blocklenghts,displacements,types,&image_dt);
	MPI_Type_commit(&image_dt);

	// Create the User defined reduction function
	MPI_Op mpi_mysum; 
	MPI_Op_create(mysum, 0, &mpi_mysum);

	MPI_Reduce(&myImage, &globalImage, 1, image_dt, mpi_mysum, 0, MPI_COMM_WORLD);

	// Print the result
	if (rank == 0) {
		printf("\n%d:Image data:\n\t", rank);
		printImage(globalImage);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
