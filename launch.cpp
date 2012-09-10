/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// includes, system
#include <iostream>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

// includes, CUDA
#include <cuda.h>
#include <builtin_types.h>
#include "drvapi_error_string.h"

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}


inline CUdevice cudaDeviceInit()
{
    CUdevice cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    if (CUDA_SUCCESS == err)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(-1);
    }
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    char name[100];
    cuDeviceGetName(name, 100, cuDevice);
    printf("Using CUDA Device [0]: %s\n", name);

    int major=0, minor=0;
    checkCudaErrors( cuDeviceComputeCapability(&major, &minor, cuDevice) );
    if (major < 2) {
        fprintf(stderr, "Device 0 is not sm_20 or later\n");
        exit(-1);
    }
    return cuDevice;
}

CUresult initCUDA(const char *kernelname, 
                  CUcontext *phContext,
                  CUdevice *phDevice,
                  CUmodule *phModule,
                  CUfunction *phKernel,	
                  const char *ptx)
{
    // Initialize 
    *phDevice = cudaDeviceInit();

    // Create context on the device
    checkCudaErrors(cuCtxCreate(phContext, CU_CTX_BLOCKING_SYNC, *phDevice));

    // Load the PTX 
    checkCudaErrors(cuModuleLoadData(phModule, ptx));

    // Locate the kernel entry point
    
    checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, kernelname));

    return CUDA_SUCCESS;
}

void LaunchOnGpu(const char *kernel, 
                 unsigned funcarity, 
                 unsigned N, 
                 void **args, 
                 void *resbuf,
                 const char *ptxBuff) 
{ 
  const unsigned int nThreads = N;
  CUcontext    hContext = 0;
  CUdevice     hDevice  = 0;
  CUmodule     hModule  = 0;
  CUfunction   hKernel  = 0;
  CUdeviceptr  d_data   = 0;
  double       *h_data   = 0;
  unsigned int nBlocks  = 1;

  // Initialize the device and get a handle to the kernel
  checkCudaErrors(initCUDA(kernel, &hContext, &hDevice, &hModule, &hKernel, ptxBuff));

  // Allocate memory for result vector on the host and device
  h_data = (double *)resbuf;
  unsigned i; 
  CUdeviceptr *deviceargs = (CUdeviceptr*) malloc(sizeof(CUdeviceptr)*funcarity);
  for (i = 0; i < funcarity; i++) { 
    double *argi = (double *) args[i];
    checkCudaErrors(cuMemAlloc(&deviceargs[i], N*sizeof(double)));
    checkCudaErrors(cuMemcpyHtoD(deviceargs[i], argi, N*sizeof(double)));
  }

  checkCudaErrors(cuFuncSetBlockShape(hKernel, nThreads, 1, 1));
  // Set the kernel parameters
  int paramOffset = 0;
  for (i = 0; i < funcarity; i++) { 
    checkCudaErrors(cuParamSetv(hKernel, paramOffset, &deviceargs[i], sizeof(deviceargs[i])));
    paramOffset += sizeof(deviceargs[i]);
  }

  checkCudaErrors(cuMemAlloc(&d_data, N*sizeof(double)));
  // pass the device ptr for result as the last argument. 
  checkCudaErrors(cuParamSetv(hKernel, paramOffset, &d_data, sizeof(d_data)));

  checkCudaErrors(cuParamSetSize(hKernel, paramOffset+sizeof(d_data)));
		   
  // Launch the kernel
  checkCudaErrors(cuLaunchGrid(hKernel, nBlocks, 1));

  checkCudaErrors(cuCtxSynchronize());
    
  // Copy the result back to the host
  checkCudaErrors(cuMemcpyDtoH(h_data, d_data, N*sizeof(double)));

  // free the allocated memory for the arguments 
  for (i = 0; i < funcarity; i++) { 
    double *argi = (double *) args[i];
    checkCudaErrors(cuMemFree(deviceargs[i]));
  }
  checkCudaErrors(cuMemFree(d_data));
  checkCudaErrors(cuModuleUnload(hModule));
  checkCudaErrors(cuCtxDestroy(hContext));
}
