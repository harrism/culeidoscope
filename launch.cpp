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

#include <algorithm>

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
    {
        const unsigned int jitNumOptions = 2;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void*[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // compile with set parameters
        CUresult status = cuModuleLoadDataEx(phModule, ptx, jitNumOptions, jitOptions, (void **)jitOptVals);

        if (CUDA_SUCCESS != status)
          printf("> PTX JIT log:\n%s\n", jitLogBuffer);
        
        delete [] jitOptions;
        delete [] jitOptVals;
        delete [] jitLogBuffer;

        checkCudaErrors(status);
    }
    
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
  const unsigned int nThreads = std::min<unsigned>(N, 128);
  const unsigned int nBlocks = (N + nThreads - 1) / nThreads;
  CUcontext    hContext = 0;
  CUdevice     hDevice  = 0;
  CUmodule     hModule  = 0;
  CUfunction   hKernel  = 0;
  CUdeviceptr  d_data   = 0;
  double       *h_data   = 0;

  // Initialize the device and get a handle to the kernel
  checkCudaErrors(initCUDA(kernel, &hContext, &hDevice, &hModule, &hKernel, ptxBuff));

  // Allocate memory for result vector on the host and device
  h_data = (double *)resbuf;
  unsigned i; 
  CUdeviceptr *deviceargs = (CUdeviceptr*) malloc(sizeof(CUdeviceptr)*(funcarity + 1));
  for (i = 0; i < funcarity; i++) { 
    double *argi = (double *) args[i];
    checkCudaErrors(cuMemAlloc(&deviceargs[i], N*sizeof(double)));
    checkCudaErrors(cuMemcpyHtoD(deviceargs[i], argi, N*sizeof(double)));
  }
  checkCudaErrors(cuMemAlloc(&deviceargs[funcarity], N*sizeof(double))); // return value

  // Set the kernel parameters
  void** params = new void*[funcarity+2];
  params[0] = (void*)&N;                // length
  for (i = 1; i < funcarity + 2; i++) { // input and output pointers
    params[i] = &deviceargs[i-1];
  }

  // Launch the kernel
  checkCudaErrors(cuLaunchKernel(hKernel, nBlocks, 1, 1, nThreads, 1, 1, 0, 0, params, 0));
  	       
  // Copy the result back to the host
  checkCudaErrors(cuMemcpyDtoH(h_data, deviceargs[funcarity], N*sizeof(double)));

  // free the allocated memory for the arguments 
  for (i = 0; i < funcarity+1; i++) { 
    checkCudaErrors(cuMemFree(deviceargs[i]));
  }
  delete [] params;
  checkCudaErrors(cuModuleUnload(hModule));
  checkCudaErrors(cuCtxDestroy(hContext));
}
