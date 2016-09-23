/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http://www.microsoft.com                 */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*                 File: HCUDA.cu:   CUDA Utilities            */
/* ----------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

char *hcuda_version = "!HVER!HCUDA:   3.4.1 [CUED 30/11/13]";
char *hcuda_vc_id = "$Id: HCUDA.cu,v 1.1.1.1 2013/11/13 09:54:58 cz277 Exp $";

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "HCUDA.h"
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "cfgs.h"


/* --------------------------- Trace Flags ------------------------ */

#define CEIL(x,y) (((x)+(y)-1) / (y))

/* --------------------------- Trace Flags ------------------------ */

static int trace = 0;                           /*  */
#define T_TOP 0001                              /* Top Level tracing */

static ConfParam *cParm[MAXGLOBS];              /* config parameters */
static int nParm = 0;

static int GPUDevId = -1;                       /*  */
static Boolean GPUInit = FALSE;                 /*  */
static char *GPUIdEnvVar = "";                  /*  */
cublasHandle_t handle;				/*  */
static size_t GPUMemUsed = 0;			/*  */

/* ----------------------- Device Management ---------------------- */

/*  */
static void ShowAllGPUs(void) {
    int nGPU, i;
    cudaError_t error;
    cudaDeviceProp prop;
    /*CUResult result;*/

    error = cudaGetDeviceCount(&nGPU);    
    if (error != cudaSuccess) {
        HError(9999, "ShowAllGPUs: %s", cudaGetErrorString(error)); 
    }
    if (nGPU == 0) {
        HError(9999, "ShowAllGPUs: No GPU device");
    }
    for (i = 0; i < nGPU; ++i) {
        error = cudaGetDeviceProperties(&prop, i);
        if (error != cudaSuccess) {
            HError(9999, "ShowAllGPUs: %s", cudaGetErrorString(error));
        }
        printf("GPU %d: %s, %dMB, SM = %d.%d", i, prop.name, prop.totalGlobalMem / 1048576, prop.major, prop.minor);
        if (GPUDevId == i)
            printf(" [Selected]");
        printf("\n");
    }
}

/* To check CUDA requirement */
static void CheckCUDAReq(cudaDeviceProp *prop)
{
    int driverVer;
    int runtimeVer;
    int cublasVer;
    cudaError_t error;    
    cublasStatus_t status;
    
    error = cudaDriverGetVersion(&driverVer);
    if (error != cudaSuccess) {
        HError(9999, "CheckCUDAReq: %s", cudaGetErrorString(error));
    }
    if (driverVer < MINCUDAVER) {
        HError(9999, "CheckCUDAReq: CUDA driver version %d is lower than the minimum required version %d", driverVer, MINCUDAVER);
    }

    error = cudaRuntimeGetVersion(&runtimeVer);
    if (error != cudaSuccess) {
        HError(9999, "CheckCUDAReq: %s", cudaGetErrorString(error));
    }
    if (runtimeVer < MINCUDAVER) {
        HError(9999, "CheckCUDAReq: CUDA runtime version %d is lower than the minimum required version %d", runtimeVer, MINCUDAVER);
    }

    status = cublasGetVersion(handle, &cublasVer);
    if (status != CUBLAS_STATUS_SUCCESS) {
        HError(9999, "CheckCUDAReq: Fail to get CUBLAS library version");
    }
    if (cublasVer < MINCUDAVER) {
        HError(9999, "CheckCUDAReq: CUBLAS library version %d is lower than the minimum required version %d", cublasVer, MINCUDAVER);
    }

    if (prop->major <= MINMAJORSMARCH && prop->minor <= MINMINORSMARCH) {
        HError(9999, "CheckCUDAReq: SM architecture is lower than the minimum requirement, %d.%d", MINMAJORSMARCH, MINMINORSMARCH);
    }

    printf("CUDA driver version %d\n", driverVer);
    printf("CUDA runtime version %d\n", runtimeVer);
    printf("CUBLAS library version %d\n", cublasVer);
}

/* Initialize the GPU device. It first loads the GPU device
   from the config file. Then
*/
void InitCUDA(void)
{
    int intVal;
    char buf[256];
    ConfParam *cpVal;

    Register(hcuda_version, hcuda_vc_id);

    /* load parameters from the config file */
    nParm = GetConfig("HCUDA", TRUE, cParm, MAXGLOBS);
    if (nParm > 0) {
        if (GetConfInt(cParm, nParm, "TRACE", &intVal)) {
            trace = intVal;
        }
        if (GetConfAny(cParm, nParm, "GPUID", &cpVal)) {
            if (cpVal->kind == IntCKind) {
                GPUDevId = cpVal->val.i;
            }
            else if (cpVal->kind == StrCKind) {
                strcpy(buf, cpVal->val.s);
                GPUIdEnvVar = (char *) New(&gcheap, sizeof(char) * strlen(buf));
                strcpy(GPUIdEnvVar, buf);
            }
            else {
                HError(9999, "InitCUDA: Unknown GPUID kind");
            }
        }
    }
}

/*  */
void StartCUDA(void) {
    char *envVar;
    cudaError_t error;
    cublasStatus_t status;
    cudaDeviceProp prop;

    /* initialize the library and device */
    if (!GPUInit) {
        /* select a device */
        if (strcmp(GPUIdEnvVar, "") != 0) { /* use env variable */
            envVar = getenv(GPUIdEnvVar);
            if (envVar == NULL) {
                printf("InitCUDA: Fail to get environment variable %s\n", GPUIdEnvVar);
            }
            GPUDevId =  atoi(envVar);
        }
        if (GPUDevId < 0) {
            error = cudaChooseDevice(&GPUDevId, &prop);
            if (error != cudaSuccess) {
                HError(9999, "InitCUDA: %s", cudaGetErrorString(error));
            }
        }
        error = cudaSetDevice(GPUDevId);
        if (error != cudaSuccess) {
            HError(9999, "InitCUDA: %s", cudaGetErrorString(error));
        }
        error = cudaGetDeviceProperties(&prop, GPUDevId);
        if (error != cudaSuccess) {
            HError(9999, "InitCUDA: %s", cudaGetErrorString(error));
        }
        /* initiate CUBLAS */
        status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            HError(9999, "InitCUDA: Fail to initialise CUBLAS");
        }
        /* check version */
        CheckCUDAReq(&prop);
        /* set GPUInit flag */
        GPUInit = TRUE;
        /* show devices */
        ShowAllGPUs();
    }
    else {
        printf("InitCUDA: GPU device %d already initialised", GPUDevId);
    }
}

/*  */
void StopCUDA(void) {
    if (GPUInit) {
        /* destroy the context on the GPU */
        cublasDestroy(handle);
        /* shutdown CUBLAS */
        cudaDeviceReset();
        /* reset GPU IDs and the flag */
        GPUDevId = -1;
        GPUInit = FALSE;
    }
    else {
        printf("StopCUDA: GPU device has already stopped");
    }
}

/* --------------------------- Trace Flags ------------------------ */

__global__ void HKern_SetNSegment(NFloat val, NFloat *segPtr, int segLen) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        segPtr[pos] = val;
    }
}

__global__ void HKern_ScaledSelfAddNSegment(NFloat *rhPtr, int segLen, NFloat scale, NFloat *lhPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        lhPtr[pos] = scale * lhPtr[pos] + rhPtr[pos];
    }
}

__global__ void HKern_DupNSegment(NFloat *srcPtr, int segLen, NFloat *dstPtr, int times) {
    int srcPos, dstPos;
    
    dstPos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (dstPos < segLen * times) {
        srcPos = dstPos % segLen;
        dstPtr[dstPos] = srcPtr[srcPos];
    }
}

__global__ void HKern_SubNSegment(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        resPtr[pos] = lhPtr[pos] - rhPtr[pos];
    }
}

__global__ void HKern_MulNSegment(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        resPtr[pos] = lhPtr[pos] * rhPtr[pos];
    }
}

/* cz277 - pact */
__global__ void HKern_ApplyAffineAct(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, NFloat *dstPtr) {
    int pos, rowIdx, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    rowIdx = pos / col;
    colIdx = pos % col;
    if (rowIdx < row) {
        dstPtr[pos] = scalePtr[colIdx] * srcPtr[pos] + shiftPtr[colIdx];
    }
}

/* cz277 - pact */
__global__ void HKern_ApplyDAffineAct(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, NFloat *dstPtr) {
    int pos, rowIdx, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    rowIdx = pos / col;
    colIdx = pos % col;
    if (rowIdx < row) {
        dstPtr[pos] = scalePtr[colIdx];
    }
}


/* cz277 - pact */
__global__ void HKern_ApplyTrAffineAct(NFloat *errPtr, NFloat *actPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, Boolean accFlag, NFloat *dScalePtr, NFloat *dShiftPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step, off = THREADPERBLOCK;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;		/* dScale */
        tmpPtr[off + thdIdx] = 0.0;	/* dShift */
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            tmpPtr[thdIdx] += errPtr[pos] * actPtr[pos];
            tmpPtr[off + thdIdx] += errPtr[pos];
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                    tmpPtr[off + thdIdx] += tmpPtr[off + pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            if (accFlag == FALSE) {
                dScalePtr[colIdx] = 0.0;
                dShiftPtr[colIdx] = 0.0;
            }
            dScalePtr[colIdx] += tmpPtr[0];
            dShiftPtr[colIdx] += tmpPtr[off + 0];
        }
    }
}

/* cz277 - laf */
__global__ void HKern_AccMeanNSegment(NFloat *valPtr, int row, int col, NFloat tSamp, NFloat *meanPtr) {
        extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        /*base = colIdx;*/
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;/*srcPtr[base + idx * col];*/
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            tmpPtr[thdIdx] += valPtr[pos] / tSamp;
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            meanPtr[colIdx] += tmpPtr[0];
        }
    }
}

/* cz277 - laf */
__global__ void HKern_AccVarianceNSegment(NFloat *valPtr, int row, int col, NFloat tSamp, NFloat *meanPtr, NFloat *varPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        /*base = colIdx;*/
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;/*srcPtr[base + idx * col];*/
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            tmpPtr[thdIdx] += pow(valPtr[pos] - meanPtr[colIdx], 2) / tSamp;
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            varPtr[colIdx] += tmpPtr[0];
        }
    }
}


/* cz277 - pact */
__global__ void HKern_ApplyParmReLUAct(NFloat *srcPtr, int row, int col, NFloat *posPtr, NFloat *negPtr, NFloat *dstPtr) {
    int pos, rowIdx, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    rowIdx = pos / col;
    colIdx = pos % col;
    if (rowIdx < row) {
        if (srcPtr[pos] > 0.0)
            dstPtr[pos] = posPtr[colIdx] * srcPtr[pos];
        else
            dstPtr[pos] = negPtr[colIdx] * srcPtr[pos];
    }
}

/* cz277 - pact */
__global__ void HKern_ApplyDParmReLUAct(NFloat *inpPtr, int row, int col, NFloat *posPtr, NFloat *negPtr, NFloat *dstPtr) {
    int pos, rowIdx, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    rowIdx = pos / col;
    colIdx = pos % col;
    if (rowIdx < row) {
        if (inpPtr[pos] > 0.0)
            dstPtr[pos] = posPtr[colIdx];
        else
            dstPtr[pos] = negPtr[colIdx];
    }
}


/* cz277 - pact */
__global__ void HKern_ApplyTrParmReLUAct(NFloat *errPtr, NFloat *inpPtr, int row, int col, Boolean accFlag, NFloat *dPosPtr, NFloat *dNegPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step, off = THREADPERBLOCK;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;		/* alpha */
        tmpPtr[off + thdIdx] = 0.0;	/* beta */
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            if (inpPtr[pos] > 0.0)
                tmpPtr[thdIdx] += errPtr[pos] * inpPtr[pos];
            else
                tmpPtr[off + thdIdx] += errPtr[pos] * inpPtr[pos];
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                    tmpPtr[off + thdIdx] += tmpPtr[off + pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            if (accFlag == FALSE) {
                dPosPtr[colIdx] = 0.0;
                dNegPtr[colIdx] = 0.0;
            }
            dPosPtr[colIdx] += tmpPtr[0];
            dNegPtr[colIdx] += tmpPtr[off + 0];
        }
    }
}


/* cz277 - laf */
__global__ void HKern_ApplyPReLUAct(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *dstPtr) {
    int pos, rowIdx, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    rowIdx = pos / col;
    colIdx = pos % col;
    if (rowIdx < row) {
        if (srcPtr[pos] > 0.0)
            dstPtr[pos] = scalePtr[colIdx] * srcPtr[pos];
        else
            dstPtr[pos] = 0.0;
    }
}

/* cz277 - pact */
__global__ void HKern_ApplyDPReLUAct(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *dstPtr) {
    int pos, rowIdx, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    rowIdx = pos / col;
    colIdx = pos % col;
    if (rowIdx < row) {
        if (scalePtr[colIdx] != 0.0 && srcPtr[pos] / scalePtr[colIdx] > 0.0)
            dstPtr[pos] = scalePtr[colIdx];
        else
            dstPtr[pos] = 0.0;
    }
}

/* cz277 - pact */
__global__ void HKern_ApplyTrPReLUAct(NFloat *errPtr, NFloat *srcPtr, int row, int col, NFloat *scalePtr, Boolean accFlag, NFloat *dScalePtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step;
    NFloat act;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;	/*srcPtr[base + idx * col];*/
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            if (scalePtr[colIdx] != 0.0) {
                act = srcPtr[pos] / scalePtr[colIdx];
                if (act > 0.0)
                    tmpPtr[thdIdx] += errPtr[pos] * act;
            }
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            if (accFlag == FALSE)
                dScalePtr[colIdx] = 0.0;
            dScalePtr[colIdx] += tmpPtr[0];
        }
    }
}

__global__ void HKern_ApplyReLUAct(NFloat *srcPtr, int len, NFloat scale, NFloat *dstPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        if (srcPtr != dstPtr && srcPtr[pos] > 0) {
            dstPtr[pos] = srcPtr[pos];
        }
        if (srcPtr[pos] < 0) {
            dstPtr[pos] = srcPtr[pos] * scale;
            /* cz277 - standard ReLU */
            /*dstPtr[pos] = 0.0;*/
        }
    }
}

__global__ void HKern_ApplyDReLUAct(NFloat *srcPtr, int len, NFloat scale, NFloat *dstPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        if (srcPtr[pos] > 0.0) {
            dstPtr[pos] = 1.0;
        }
        else {
            dstPtr[pos] = scale;
            /* cz277 - standard ReLU */
            /*dstPtr[pos] = 0.0;*/
        }
    }
}

__global__ void HKern_ApplyDLinearAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        dstPtr[pos] = 1.0;
    }
}

__global__ void HKern_ApplyLHUCSigmoidAct(NFloat *srcPtr, int row, int col, NFloat *rolePtr, NFloat *dstPtr) {
    int pos, colIdx;
    NFloat floatVal, lhucVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < row * col) {
        colIdx = pos % col;
        floatVal = -1.0 * rolePtr[colIdx];
        CHKNFLTEXPE(floatVal)
        lhucVal = 2.0 / (1.0 + exp(floatVal));
        floatVal = -1.0 * srcPtr[pos];
        CHKNFLTEXPE(floatVal)
        dstPtr[pos] = lhucVal * 1.0 / (1.0 + exp(floatVal));
    }
}

__global__ void HKern_ApplyDLHUCSigmoidAct(NFloat *srcPtr, int row, int col, NFloat *rolePtr, NFloat *dstPtr) {
    int pos, colIdx;
    NFloat floatVal, lhucVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < row * col) {
        colIdx = pos % col;
        floatVal = -1.0 * rolePtr[colIdx];
        CHKNFLTEXPE(floatVal)
        lhucVal = 2.0 / (1.0 + exp(floatVal));
        floatVal = srcPtr[pos] / lhucVal;
        dstPtr[pos] = srcPtr[pos] * (1.0 - floatVal);
    }
}

__global__ void HKern_ApplyTrLHUCSigmoidActCUDA(NFloat *errPtr, NFloat *actPtr, int row, int col, NFloat *rolePtr, Boolean accFlag, NFloat *dRolePtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step;
    NFloat floatVal;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        floatVal = -1.0 * rolePtr[colIdx];
        CHKNFLTEXPE(floatVal)
        floatVal = 0.5 * 2.0 / (1.0 + exp(floatVal));
        /*base = colIdx;*/
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;/*actPtr[base + idx * col];*/
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            tmpPtr[thdIdx] += errPtr[pos] * actPtr[pos] * (1.0 - floatVal);
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            if (accFlag == FALSE)
                dRolePtr[colIdx] = 0.0;
            dRolePtr[colIdx] += tmpPtr[0];
        }
    }
}


__global__ void HKern_ApplyParmSigmoidAct(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat* thetaPtr, NFloat *dstPtr) {
    int pos, colIdx;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < row * col) {
        colIdx = pos % col;
        floatVal = (-1.0) * gammaPtr[colIdx] * srcPtr[pos] + thetaPtr[colIdx];
        CHKNFLTEXPE(floatVal)
        dstPtr[pos] = etaPtr[colIdx] / (1.0 + exp(floatVal));
    }
}

__global__ void HKern_ApplyDParmSigmoidAct(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat *thetaPtr, NFloat *dstPtr) {
    int pos, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < row * col) {
        colIdx = pos % col;
        if (etaPtr[colIdx] != 0.0)
            dstPtr[pos] = gammaPtr[colIdx] * srcPtr[pos] * (1.0 - srcPtr[pos] / etaPtr[colIdx]);
        else
            dstPtr[pos] = 0.0;
    }
}

__global__ void HKern_ApplyTrParmSigmoidActCUDA(NFloat *errPtr, NFloat *inpPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat *thetaPtr, Boolean accFlag, NFloat *dEtaPtr, NFloat *dGammaPtr, NFloat *dThetaPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step, off = THREADPERBLOCK;
    NFloat floatVal, fracVal;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        /*base = colIdx;*/
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;	/*actPtr[base + idx * col];*/
        tmpPtr[off + thdIdx] = 0.0;
        tmpPtr[off + off + thdIdx] = 0.0;
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            floatVal = (-1.0) * gammaPtr[colIdx] * inpPtr[pos] + thetaPtr[colIdx];
            CHKNFLTEXPE(floatVal)
            fracVal = 1.0 / (1.0 + exp(floatVal));
            tmpPtr[thdIdx] += errPtr[pos] * fracVal;
            if (etaPtr[colIdx] != 0.0) {
                tmpPtr[off + thdIdx] += errPtr[pos] * inpPtr[pos] * etaPtr[colIdx] * fracVal * (1.0 - fracVal);
                tmpPtr[off + off + thdIdx] -= errPtr[pos] * etaPtr[colIdx] * fracVal * (1.0 - fracVal);
            }  
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                    tmpPtr[off + thdIdx] += tmpPtr[off + pos];
                    tmpPtr[off + off + thdIdx] += tmpPtr[off + off + pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            if (accFlag == FALSE) {
                dEtaPtr[colIdx] = 0.0;
                dGammaPtr[colIdx] = 0.0;
                dThetaPtr[colIdx] = 0.0;
            }
            dEtaPtr[colIdx] += tmpPtr[0];
            dGammaPtr[colIdx] += tmpPtr[off + 0];
            dThetaPtr[colIdx] += tmpPtr[off + off + 0];
        }
    }
}


__global__ void HKern_ApplyPSigmoidAct(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *dstPtr) {
    int pos, colIdx;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < row * col) {
        colIdx = pos % col;
        floatVal = (-1.0) * srcPtr[pos];
        CHKNFLTEXPE(floatVal)
        dstPtr[pos] = etaPtr[colIdx] / (1.0 + exp(floatVal));
    }
}

__global__ void HKern_ApplyDPSigmoidAct(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *dstPtr) {
    int pos, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < row * col) {
        colIdx = pos % col;
        /* dstPtr[pos] = srcPtr[pos] * (1.0 - srcPtr[pos] / etaPtr[colIdx]); */
        if (etaPtr[colIdx] != 0.0)
            dstPtr[pos] = 1.0 / etaPtr[colIdx] * srcPtr[pos] * (etaPtr[colIdx] - srcPtr[pos]);
        else
            dstPtr[pos] = 0.0;
    }
}

__global__ void HKern_ApplyTrPSigmoidActCUDA(NFloat *errPtr, NFloat *srcPtr, NFloat *etaPtr, int row, int col, Boolean accFlag, NFloat *dEtaPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        /*base = colIdx;*/
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;	/*actPtr[base + idx * col];*/
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            /* tmpPtr[thdIdx] += errPtr[pos] * srcPtr[pos] / etaPtr[colIdx]; */
            if (etaPtr[colIdx] != 0.0)
                tmpPtr[thdIdx] += errPtr[pos] * 1.0 / etaPtr[colIdx] * srcPtr[pos];
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            if (accFlag == FALSE)
                dEtaPtr[colIdx] = 0.0;
            dEtaPtr[colIdx] += tmpPtr[0];
        }
    }
}


__global__ void HKern_ApplySigmoidAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = -1.0 * srcPtr[pos];
        CHKNFLTEXPE(floatVal)
        dstPtr[pos] = 1.0 / (1.0 + exp(floatVal));
    }
}

__global__ void HKern_ApplyDSigmoidAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        dstPtr[pos] = (1 - srcPtr[pos]) * srcPtr[pos];
    }
}

__global__ void HKern_ApplyTanHAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = srcPtr[pos];
        CHKNFLTEXPE(floatVal)
        floatVal = exp(floatVal);
        dstPtr[pos] = (floatVal - 1.0 / floatVal) / (floatVal + 1.0 / floatVal);
    }
}

__global__ void HKern_ApplyDTanHAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        dstPtr[pos] = 1 - pow(srcPtr[pos], 2);
    }
}

__global__ void HKern_DualSumByRow(NFloat *srcPtr, int col, int size, int incr, NFloat *dstPtr) {
    int lhpos, rhpos, lhidx, rhidx, mod;

    lhpos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (lhpos < size) {
        mod = incr * 2;
        lhidx = lhpos % col;
        if (lhidx % mod == 0) {
            rhidx = lhidx + incr;
            rhpos = lhpos + incr;
            if (rhidx >= col) {
                dstPtr[lhpos] = srcPtr[lhpos];
            }
            else {
                dstPtr[lhpos] = srcPtr[lhpos] + srcPtr[rhpos];
            }
        }
    }
}

__global__ void HKern_ApplySoftmaxAct(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int frame, i, base, off;
    NFloat den, floatVal;

    frame = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (frame < row) {
        den = 0.0;
        base = frame * col;
        for (i = 0, off = base; i < col; ++i, ++off) {
            floatVal = srcPtr[off];
            CHKNFLTEXPE(floatVal)
            floatVal = exp(floatVal);
            dstPtr[off] = floatVal;
            den += floatVal;
        }
        for (i = 0, off = base; i < col; ++i, ++off) {
            dstPtr[off] /= den;
        }
    }
}


//cw564 - stimu -- begin
#define LITTLE 0.0000001
#define RESVAR 9.5
#define SCALERVAR 100.0
#define COMBLITTLE 1e-6

__device__ inline NFloat calc_tanh(NFloat raw) {
    float floatVal = raw;
    CHKNFLTEXPE(floatVal)
    floatVal = exp(floatVal);
    return (floatVal - 1.0 / floatVal) / (floatVal + 1.0 / floatVal);
}

__device__ inline NFloat calc_exp(NFloat raw) {
    CHKNFLTEXPE(raw)
    return exp(raw);
}


__global__ void HKern_MixValidate(NFloat *act, int num_phones, int len) {
    int myidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (myidx >= 2 * num_phones) {
        return;
    }
    int my_ptr = myidx / 2 * 6 + myidx % 2;
    if (act[my_ptr] > 1.0) {
        act[my_ptr] = 1.0;
    }
    else if (act[my_ptr] < 0.0){
        act[my_ptr] = 0.0;
    }
    /*if (myidx >= len) return;
    int my_dim = myidx % 5;
    if (my_dim == 4) {
        my_dim = 5;    
    }
    int my_phone = myidx / 5;
    int my_ptr = my_dim + my_phone * 6;
    if (my_dim == 0 || my_dim == 1) {
        return;
    }
    else if (my_dim == 3) {
        if (act[my_ptr] <= -1.0) {
            act[my_ptr] = -0.999999;
        }
        else if (act[my_ptr] >= 1.0) {
            act[my_ptr] = 0.999999;
        }
    }
    else if (my_dim == 2 || my_dim == 5) {
        if (act[my_ptr] <= 0) {
            act[my_ptr] = 0.000001;
        }
    }*/
}

void MixValidateCUDA(NFloat *act, int num_phones, int len) {
    int blocks = CEIL(2 * num_phones, THREADPERBLOCK);
    //printf("%d %d\n", blocks, num_phones);exit(0);
    HKern_MixValidate<<< blocks, THREADPERBLOCK >>>(act, num_phones, len);
}


__global__ void HKern_CalcMixSurface(NFloat* mix_surface, NFloat* actParmVec, int comb_dim, int mixnodes, float grid_var, int tot, int grid_one_dim) {

    int i, q, r, row_no, col_no;
    float x, y, norm, s1, s2, rou, s1s2, one_min_rousqr;
    float diff1, diff2, diff;
    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < tot) {
        q = i / comb_dim;
        r = i % comb_dim;
        row_no = r / grid_one_dim;
        col_no = r % grid_one_dim;
        x = (0.5 + col_no) / grid_one_dim;
        y = (0.5 + row_no) / grid_one_dim;
    
        s1 = calc_exp(actParmVec[q * 6 + 2]);
        s2 = calc_exp(actParmVec[q * 6 + 5]);
        rou = calc_tanh(actParmVec[q * 6 + 3]);

        //s1 = actParmVec[q * 6 + 2];
        //s2 = actParmVec[q * 6 + 5];
        //rou = actParmVec[q * 6 + 3];

        s1s2 = s1 * s2;

        one_min_rousqr = 1 - rou * rou;
        norm = 1.0 / (PI * 2.0 * sqrtf(one_min_rousqr) * s1s2);

        diff1 = x - actParmVec[q * 6];
        diff2 = y - actParmVec[q * 6 + 1];
        diff = -(powf(diff1 / s1, 2) + powf(diff2 / s2, 2) - 2 * rou * diff1 * diff2 / s1s2) / (2 * one_min_rousqr);
        
        mix_surface[i] = exp(diff) * norm;
    }
}

void CalcMixSurfaceCUDA(NFloat* mix_surface, NFloat* actParmVec, int comb_dim, int mixnodes, float grid_var) {
    int tot = comb_dim * mixnodes;
    //printf("%d %d\n", comb_dim, mixnodes);
    int nBlocks = CEIL(tot, THREADPERBLOCK);
    int grid_one_dim = (int)sqrtf(comb_dim);
    HKern_CalcMixSurface<<< nBlocks, THREADPERBLOCK >>>(mix_surface, actParmVec, comb_dim, mixnodes, grid_var, tot, grid_one_dim);
}

__global__ void HKern_ResetZero(NFloat *dact, int len) {
    int my = blockIdx.x * blockDim.x + threadIdx.x;
    if (my >= len) return;
    dact[my] = 0.0;
}

__global__ void HKern_ApplyGradMixRels(NFloat* dact, NFloat* act, NFloat* dyFeaMat, NFloat* mix_surface, NFloat * raw_y, int batLen, int nodeNum, int comb_dim, int mixnodes, NFloat mix_rel_l2_penalty) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, batIdx, mixIdx, mixdimIdx, thdNum, dybase, mixbase, idx, incr, pos, nodeIdx;
    int total_jobs = batLen * comb_dim;
    NFloat sumVal, tmpVal, x, y, s1, s2, rou, m1, m2;
    int row_no, col_no;
    thdIdx = threadIdx.x;
    mixIdx = blockIdx.x;
    mixdimIdx = 3;
    thdNum = min(blockDim.x, total_jobs);
    if (thdIdx < thdNum && mixIdx < mixnodes) { // && (mixIdx == 1 || mixIdx  == 10)) {
        idx = thdIdx;
        tmpPtr[thdIdx] = 0;

        m1 = act[mixIdx * 6 + 0];
        m2 = act[mixIdx * 6 + 1];
        s1 = calc_exp(act[mixIdx * 6 + 2]);
        rou= calc_tanh(act[mixIdx * 6 + 3]);
        s2 = calc_exp(act[mixIdx * 6 + 5]);
        while (idx < total_jobs) {
            batIdx = idx / comb_dim;
            nodeIdx = idx % comb_dim;
            row_no = nodeIdx / 32;
            col_no = nodeIdx % 32;

            x = (0.5 + col_no) / 32.0;
            y = (0.5 + row_no) / 32.0;
            tmpVal =  rou / powf(1-rou, 2) + 
                    (y * rou * s1 - x * s2 - rou * m2 * s1 + m1 * s2) * (-y * s1 + x * rou * s2 - rou * m1 * s2 + m2 * s1) / (powf(s1, 2) * powf(s2, 2) * powf(rou - 1, 2) * powf(rou + 1, 2));
                            //scaler                                        * mixweights                                  * comb_dy
            tmpPtr[thdIdx] += raw_y[batIdx * nodeNum + comb_dim + mixnodes] * raw_y[batIdx * nodeNum + comb_dim + mixIdx] * dyFeaMat[batIdx * comb_dim + nodeIdx] 
                    //mix_dist
                    * mix_surface[mixIdx * comb_dim + nodeIdx] * tmpVal;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++ incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        if (thdIdx == 0) {
            //dact[mixIdx * 6 + mixdimIdx] = 2.0 * sumVal / comb_dim * 10.0;
            dact[mixIdx * 6 + mixdimIdx] = sumVal * (1 - powf(rou, 2)) + act[mixIdx * 6 + mixdimIdx] * mix_rel_l2_penalty;
            //dact[mixIdx * 6 + mixdimIdx] = 0;
        }
    }
}

void ApplyGradMixRelsCUDA(NFloat* dact, NFloat* act, NFloat* dyFeaMat, NFloat* mix_surface, NFloat* raw_y, int batLen, int nodeNum, int comb_dim, int mixnodes, NFloat mix_rel_l2_penalty) {
    int blocks = mixnodes;
    //int threads = batLen;
    int sbytes = THREADPERBLOCK * sizeof(float);
    int zblocks = CEIL(comb_dim, THREADPERBLOCK);
    //HKern_ResetZero<<< zblocks, THREADPERBLOCK >>>(dact, comb_dim);
    HKern_ApplyGradMixRels<<< blocks, THREADPERBLOCK, sbytes >>>(dact, act, dyFeaMat, mix_surface, raw_y, batLen, nodeNum, comb_dim, mixnodes, mix_rel_l2_penalty);
}



__global__ void HKern_ApplyGradMixVars(NFloat* dact, NFloat* act, NFloat* dyFeaMat, NFloat* mix_surface, NFloat * raw_y, int batLen, int nodeNum, int comb_dim, int mixnodes, NFloat mix_var_l2_penalty, NFloat gridvar) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, batIdx, mixIdx, mixdimIdx, thdNum, dybase, mixbase, idx, incr, pos, nodeIdx;
    int total_jobs = batLen * comb_dim;
    NFloat sumVal, tmpVal, x, y, s1, s2, rou, m1, m2;
    int row_no, col_no;
    thdIdx = threadIdx.x;
    mixIdx = blockIdx.x / 2;
    mixdimIdx = blockIdx.x % 2;
    if (mixdimIdx) {
        mixdimIdx = 5; //second var
    } else {
        mixdimIdx = 2; //first var
    }
    thdNum = min(blockDim.x, total_jobs);
    if (thdIdx < thdNum && mixIdx < mixnodes) { // && (mixIdx == 1 || mixIdx  == 10)) {
        idx = thdIdx;
        tmpPtr[thdIdx] = 0;

        m1 = act[mixIdx * 6 + 0];
        m2 = act[mixIdx * 6 + 1];
        s1 = calc_exp(act[mixIdx * 6 + 2]);
        rou= calc_tanh(act[mixIdx * 6 + 3]);
        s2 = calc_exp(act[mixIdx * 6 + 5]);
        while (idx < total_jobs) {
            batIdx = idx / comb_dim;
            nodeIdx = idx % comb_dim;
            row_no = nodeIdx / 32;
            col_no = nodeIdx % 32;

            x = (0.5 + col_no) / 32.0;
            y = (0.5 + row_no) / 32.0;
            if (mixdimIdx == 2) {
                //tmpVal =  -1.0 / (s1) + (x - m1) * (y * rou * s1 + s2 * m1 - s2 * x - rou * m2 * s1) / ((powf(rou, 2) - 1) * powf(s1, 3) * s2);
                tmpVal =  -1.0 / (s1) + (m1 - x) * (-m1 * s2 + m2 * rou * s1 - rou * s1 * y + s2 * x) / ((powf(rou, 2) - 1) * powf(s1, 3) * s2);
                                //scaler                                        * mixweights                                  * comb_dy
                tmpPtr[thdIdx] += raw_y[batIdx * nodeNum + comb_dim + mixnodes] * raw_y[batIdx * nodeNum + comb_dim + mixIdx] * dyFeaMat[batIdx * comb_dim + nodeIdx] 
                        //mix_dist
                        * mix_surface[mixIdx * comb_dim + nodeIdx] * tmpVal;
            }
            else {
                //tmpVal =  -1.0 / (s2) + (y - m2) * (x * rou * s2 + s1 * m2 - s1 * y - rou * m1 * s2) / ((powf(rou, 2) - 1) * powf(s2, 3) * s1);
                tmpVal =  -1.0 / (s2) + (m2 - y) * (-m2 * s1 + m2 * rou * s1 - rou * s2 * x + s1 * y) / ((powf(rou, 2) - 1) * powf(s2, 3) * s1);
                tmpPtr[thdIdx] += raw_y[batIdx * nodeNum + comb_dim + mixnodes] * raw_y[batIdx * nodeNum + comb_dim + mixIdx] * dyFeaMat[batIdx * comb_dim + nodeIdx]
                        * mix_surface[mixIdx * comb_dim + nodeIdx] * tmpVal;
            }
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++ incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        if (thdIdx == 0) {
            //dact[mixIdx * 6 + mixdimIdx] = 2.0 * sumVal / comb_dim * 10.0;
            dact[mixIdx * 6 + mixdimIdx] = sumVal * calc_exp(act[mixIdx * 6 + mixdimIdx]) + (act[mixIdx * 6 + mixdimIdx] - log(sqrtf(gridvar))) * mix_var_l2_penalty;
            //dact[mixIdx * 6 + mixdimIdx] = 0;
        }
    }
}

void ApplyGradMixVarsCUDA(NFloat* dact, NFloat* act, NFloat* dyFeaMat, NFloat* mix_surface, NFloat* raw_y, int batLen, int nodeNum, int comb_dim, int mixnodes, NFloat mix_var_l2_penalty, NFloat gridvar) {
    int blocks = mixnodes * 2;
    //int threads = batLen;
    int sbytes = THREADPERBLOCK * sizeof(float);
    int zblocks = CEIL(comb_dim, THREADPERBLOCK);
    //HKern_ResetZero<<< zblocks, THREADPERBLOCK >>>(dact, comb_dim);
    HKern_ApplyGradMixVars<<< blocks, THREADPERBLOCK, sbytes >>>(dact, act, dyFeaMat, mix_surface, raw_y, batLen, nodeNum, comb_dim, mixnodes, mix_var_l2_penalty, gridvar);
}




__global__ void HKern_ApplyGradMixMeans(NFloat* dact, NFloat* act, NFloat* dyFeaMat, NFloat* mix_surface, NFloat * raw_y, int batLen, int nodeNum, int comb_dim, int mixnodes, NFloat mix_mean_l2_penalty, NFloat* phonepos) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, batIdx, mixIdx, mixdimIdx, thdNum, dybase, mixbase, idx, incr, pos, nodeIdx;
    int total_jobs = batLen * comb_dim;
    NFloat sumVal, tmpVal, x, y, s1, s2, rou, m1, m2;
    int row_no, col_no;
    thdIdx = threadIdx.x;
    mixIdx = blockIdx.x / 2;
    mixdimIdx = blockIdx.x % 2;
    thdNum = min(blockDim.x, total_jobs);
    if (thdIdx < thdNum && mixIdx < mixnodes) { // && (mixIdx == 1 || mixIdx  == 10)) {
        idx = thdIdx;
        tmpPtr[thdIdx] = 0;

        m1 = act[mixIdx * 6 + 0];
        m2 = act[mixIdx * 6 + 1];
        s1 = calc_exp(act[mixIdx * 6 + 2]);
        rou= calc_tanh(act[mixIdx * 6 + 3]);
        s2 = calc_exp(act[mixIdx * 6 + 5]);
        //s1 = act[mixIdx * 6 + 2];
        //rou= act[mixIdx * 6 + 3];
        //s2 = act[mixIdx * 6 + 5];
        while (idx < total_jobs) {
            batIdx = idx / comb_dim;
            nodeIdx = idx % comb_dim;
            row_no = nodeIdx / 32;
            col_no = nodeIdx % 32;

            x = (0.5 + col_no) / 32.0;
            y = (0.5 + row_no) / 32.0;
            if (!mixdimIdx) {
                tmpVal = (m1 * s2 - m2 * rou * s1 + rou * s1 * y - s2 * x) / ((powf(rou, 2) - 1) * powf(s1, 2) * s2);
                                //scaler                                        * mixweights                                  * comb_dy
                tmpPtr[thdIdx] += raw_y[batIdx * nodeNum + comb_dim + mixnodes] * raw_y[batIdx * nodeNum + comb_dim + mixIdx] * dyFeaMat[batIdx * comb_dim + nodeIdx] 
                        //mix_dist
                        * mix_surface[mixIdx * comb_dim + nodeIdx] * tmpVal;// (((x - m1) / powf(s1, 2) - rou * (y - m2) / s1 / s2) / (1 - powf(rou, 2)));
            }
            else {
                tmpVal = (-m1 * rou * s2 + m2 * s1 + rou * s2 * x - s1 * y) / ((powf(rou, 2) - 1) * powf(s2, 2) * s1);
                tmpPtr[thdIdx] += raw_y[batIdx * nodeNum + comb_dim + mixnodes] * raw_y[batIdx * nodeNum + comb_dim + mixIdx] * dyFeaMat[batIdx * comb_dim + nodeIdx]
                        * mix_surface[mixIdx * comb_dim + nodeIdx] * tmpVal;//(((y - m2) / powf(s2, 2) - rou * (x - m1) / s1 / s2) / (1 - powf(rou, 2)));
            }
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++ incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        if (thdIdx == 0) {
            //dact[mixIdx * 6 + mixdimIdx] = 2.0 * sumVal / comb_dim * 10.0;
            dact[mixIdx * 6 + mixdimIdx] = sumVal + (act[mixIdx * 6 + mixdimIdx] - phonepos[mixIdx * 2 + mixdimIdx]) * mix_mean_l2_penalty;
            //dact[mixIdx * 6 + mixdimIdx] = 0;
        }
    }
}

void ApplyGradMixMeansCUDA(NFloat* dact, NFloat* act, NFloat* dyFeaMat, NFloat* mix_surface, NFloat* raw_y, int batLen, int nodeNum, int comb_dim, int mixnodes, NFloat mix_mean_l2_penalty, NFloat* phonepos) {
    int blocks = mixnodes * 2;
    //int threads = batLen;
    int sbytes = THREADPERBLOCK * sizeof(float);
    int zblocks = CEIL(comb_dim, THREADPERBLOCK);
    //HKern_ResetZero<<< zblocks, THREADPERBLOCK >>>(dact, comb_dim);
    HKern_ApplyGradMixMeans<<< blocks, THREADPERBLOCK, sbytes >>>(dact, act, dyFeaMat, mix_surface, raw_y, batLen, nodeNum, comb_dim, mixnodes, mix_mean_l2_penalty, phonepos);
}

__global__ void HKern_ScaleSil(NFloat *dest_y, NFloat *labMat, NFloat scaler, int scalerid, int batLen, int nodeNum) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < batLen * nodeNum) {
        int my_batid = idx / nodeNum;
        if (labMat[my_batid * nodeNum + scalerid] > 0.5) {
            dest_y[idx] = dest_y[idx] * scaler;
        }
    }
}
void ScaleSilCUDA(NFloat *dest_y, NFloat *labMat, NFloat scaler, int scalerid, int batLen, int nodeNum) {
    int nBlocks = CEIL(batLen * nodeNum, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ScaleSilCUDA: Block number exceeds the maximum");
    HKern_ScaleSil<<< nBlocks, THREADPERBLOCK >>>(dest_y, labMat, scaler, scalerid, batLen, nodeNum);
}

__global__ void HKern_ApplyDSigmoidActStimuMix(NFloat *srcMat, int len, NFloat *dstMat, int raw_dim, int comb_dim, int mixnodes, float resdnn_var, float mixscaler_var) {
    int pos;
    NFloat floatVal;
    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        int my_dim = pos % raw_dim;
        if (my_dim  < raw_dim - mixnodes - 1) {
            //floatVal = 1.0 - srcMat[pos] * srcMat[pos];
            //dstMat[pos] = floatVal / resdnn_var;
            //dstMat[pos] = (1.0 - srcMat[pos]) * srcMat[pos];
            dstMat[pos] = 1 - pow(srcMat[pos], 2);
        }
        else if (my_dim  >= raw_dim - mixnodes - 1 && my_dim < raw_dim - 1) {
            dstMat[pos] = (1.0 - srcMat[pos]) * srcMat[pos];
        }
        else {
            dstMat[pos] = (1.0 - srcMat[pos]) * srcMat[pos] / mixscaler_var;
            //dstMat[pos] = -1.0 * srcMat[pos];
            //dstMat[pos] = (1 - srcMat[pos]) * srcMat[pos] / comb_dim;
            //dstMat[pos] = -2.0 * srcMat[pos] * sqrt(-log(srcMat[pos]));
        }
    }
}

void ApplyDSigmoidActStimuMixCUDA(NFloat *srcMat, int len, NFloat *dstMat, int raw_dim, int comb_dim, float resdnn_var, float mixscaler_var) {
    int nBlocks = CEIL(len, THREADPERBLOCK);
    int mixnodes = raw_dim - comb_dim - 1;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDSigmoidActStimuMixCUDA: Block number exceeds the maximum");
    HKern_ApplyDSigmoidActStimuMix<<< nBlocks, THREADPERBLOCK >>>(srcMat, len, dstMat, raw_dim, comb_dim, mixnodes, resdnn_var, mixscaler_var);
}


__global__ void HKern_SplitGradDNNandMix_DNNpart(NFloat *dest_dy, NFloat *comb_dy, NFloat *comb_y, int batLen, int raw_dim, int comb_dim, float mixscaler, int tot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tot) {
        return;
    }
    int col_id = idx % comb_dim;
    int bat_id = idx / comb_dim;
    //if (0.0 < comb_y[idx] && comb_y[idx] < 1.0) {
        dest_dy[bat_id * raw_dim + col_id] = mixscaler * comb_dy[idx];
    //}
    //else {
    //    dest_dy[bat_id * raw_dim + col_id] = 0;
    //}
}
__global__ void HKern_SplitGradDNNandMix_Mixpart(NFloat *dest_dy, NFloat *comb_dy, NFloat *raw_y, NFloat *mix_surface, int batLen, int raw_dim, int comb_dim, int mixnodes, float mixscaler) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, batIdx, mixIdx, thdNum, dybase, mixbase, idx, incr, pos;
    NFloat sumVal, tmpVal;
    thdIdx = threadIdx.x;
    batIdx = blockIdx.x / mixnodes;
    mixIdx = blockIdx.x % mixnodes;
    thdNum = min(blockDim.x, comb_dim);
    if (thdIdx < thdNum && batIdx < batLen && mixIdx < mixnodes) {
        dybase = batIdx * comb_dim;
        mixbase = mixIdx * comb_dim;
        idx = thdIdx;
        tmpPtr[thdIdx] = 0;
        while (idx < comb_dim) {
            tmpVal = comb_dy[dybase + idx] * mix_surface[mixbase + idx];
            tmpPtr[thdIdx] += tmpVal;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        if (thdIdx == 0) {
            dest_dy[batIdx * raw_dim + comb_dim + mixIdx] = sumVal * raw_y[batIdx * raw_dim + comb_dim + mixnodes]; // / mixnodes;
        }
    }
}
__global__ void HKern_SplitGradDNNandMix_Scalerpart(NFloat *dest_dy, NFloat *comb_dy, NFloat *raw_y, NFloat *comb_y, NFloat *mix_surface, int batLen, int raw_dim, int comb_dim, int mixnodes, float mixscaler) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, batIdx, mixIdx, thdNum, dybase, rawybase, idx, incr, pos;
    NFloat sumVal, tmpVal;
    thdIdx = threadIdx.x;
    batIdx = blockIdx.x;
    thdNum = min(blockDim.x, comb_dim);
    if (thdIdx < thdNum && batIdx < batLen) {
        dybase = batIdx * comb_dim;
        rawybase = batIdx * raw_dim;
        idx = thdIdx;
        tmpPtr[thdIdx] = 0;
        while (idx < comb_dim) {
            tmpVal = (comb_y[dybase + idx] - mixscaler * raw_y[rawybase + idx]) * comb_dy[dybase + idx];
            tmpPtr[thdIdx] += tmpVal;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        if (thdIdx == 0) {
            dest_dy[batIdx * raw_dim + comb_dim + mixnodes] = sumVal / raw_y[rawybase + comb_dim + mixnodes] / mixnodes; // / comb_dim / mixnodes;
        }
    }
}


__global__ void HKern_SplitGradDNNandMix_ZeroizeMixAndScaler(NFloat *dest_dy, int comb_dim, int raw_dim, int tot) {
    if (blockIdx.x * blockDim.x + threadIdx.x >= tot) {
        return;
    }
    int my_id = blockIdx.x * raw_dim + comb_dim + threadIdx.x;
    dest_dy[my_id] = 0.0;
}

void SplitGradDNNandMixCUDA(NFloat *dest_dy, NFloat *comb_dy, NFloat *raw_y, NFloat *comb_y, NFloat *mix_surface, int batLen, int raw_dim, int comb_dim, float mixscaler, Boolean enable_update_mix, Boolean enable_update_dnn) {
    //THREADPERBLOCK
    int tot = comb_dim * batLen;
    int nBlocks = CEIL(tot, THREADPERBLOCK);
    
    if (enable_update_dnn) {
        HKern_SplitGradDNNandMix_DNNpart<<< nBlocks, THREADPERBLOCK >>>(dest_dy, comb_dy, comb_y, batLen, raw_dim, comb_dim, mixscaler, tot);
    }
    else {
        HKern_SplitGradDNNandMix_DNNpart<<< nBlocks, THREADPERBLOCK >>>(dest_dy, comb_dy, comb_y, batLen, raw_dim, comb_dim, 0.0, tot);
    }
    
    if (!enable_update_mix) {
        int blkdim = raw_dim - comb_dim;
        nBlocks = batLen;
        HKern_SplitGradDNNandMix_ZeroizeMixAndScaler<<< nBlocks, blkdim >>>(dest_dy, comb_dim, raw_dim, blkdim * nBlocks);
        return;
    }
    else {
        int mixnodes = raw_dim - comb_dim - 1;
        nBlocks = mixnodes * batLen;
        int sBytes = THREADPERBLOCK * sizeof(NFloat);
        HKern_SplitGradDNNandMix_Mixpart<<< nBlocks, THREADPERBLOCK, sBytes >>>(dest_dy, comb_dy, raw_y, mix_surface, 
                batLen, raw_dim, comb_dim, mixnodes, mixscaler);
        //printf("comb_dim=%d,raw_dim=%d,mixnodes=%d\n", comb_dim, raw_dim, raw_dim - comb_dim - 1);
        nBlocks = batLen;
        HKern_SplitGradDNNandMix_Scalerpart<<< nBlocks, THREADPERBLOCK, sBytes >>>(dest_dy, comb_dy, raw_y, comb_y, mix_surface, 
                batLen, raw_dim, comb_dim, mixnodes, mixscaler);
    }
}

__global__ void HKern_ApplySigmoidActStimuMix(NFloat *srcPtr, int len, NFloat *dstPtr, int mixnodes, int raw_col, float resdnn_var, float mixscaler_var) {
    int pos;
    NFloat floatVal;
    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        int my_dim = pos % raw_col;
        if (my_dim  < raw_col - mixnodes - 1) {
            /*floatVal = 2.0 * srcPtr[pos] / resdnn_var;
            CHKNFLTEXPE(floatVal)
            floatVal = exp(floatVal);
            dstPtr[pos] = (floatVal - 1) / (floatVal + 1);
            */
            //floatVal = -1.0 * srcPtr[pos];
            //CHKNFLTEXPE(floatVal)
            //dstPtr[pos] = 1.0 / (1.0 + exp(floatVal));
            floatVal = srcPtr[pos];
            CHKNFLTEXPE(floatVal)
            floatVal = exp(floatVal);
            dstPtr[pos] = (floatVal - 1.0 / floatVal) / (floatVal + 1.0 / floatVal);
        }
        else if (my_dim >= raw_col - mixnodes - 1 && my_dim < raw_col - 1) {
            floatVal = srcPtr[pos];
            //CHKNFLTEXPE(floatVal)
            //dstPtr[pos] = exp(floatVal);
            dstPtr[pos] = floatVal;
        }
        else {
            //NFloat var = raw_col;
            //NFloat bias = log(mixnodes / sqrt(2.0 * PI * 0.1) + 70);
            //floatVal = -1.0 * srcPtr[pos] / var + bias;
            //CHKNFLTEXPE(floatVal)
            //dstPtr[pos] = exp(floatVal);
            //dstPtr[pos] = 1.0 / (1.0 + exp(floatVal));
            //if (dstPtr[pos] < 1e-7) {
            //    dstPtr[pos] = 1e-7;
            //}
            //floatVal = -1.0 * srcPtr[pos] * srcPtr[pos];
            //CHKNFLTEXPE(floatVal)
            //dstPtr[pos] = exp(floatVal);
            floatVal = -1.0 * srcPtr[pos] / mixscaler_var;
            CHKNFLTEXPE(floatVal)
            dstPtr[pos] = 1.0 / (1.0 + exp(floatVal));
            if (dstPtr[pos] < COMBLITTLE) {
                dstPtr[pos] = COMBLITTLE;
            }
        }
    }
}

void ApplySigmoidActStimuMixCUDA(NFloat *srcMat, int len, NFloat *dstMat, int mixnodes, int raw_col, float resdnn_var, float mixscaler_var) {
    int nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplySigmoidActStimuMixCUDA: Block number exceeds the maximum");
    //printf("mixscalervar=%f\n", mixscaler_var);
    HKern_ApplySigmoidActStimuMix<<<nBlocks, THREADPERBLOCK>>>(srcMat, len, dstMat, mixnodes, raw_col, resdnn_var, mixscaler_var);
}

__global__ void HKern_FwdCombDNNandMix(NFloat *comb_yFeaMat, NFloat *mix_surface, NFloat *raw_yFeaMat, int batLen, int raw_nodeNum, int comb_nodeNum, int mixnodes, float mixscaler) {
    extern __shared__ NFloat tmpPtr[];
    int my_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (blockIdx.x >= batLen * comb_nodeNum && threadIdx.x >= mixnodes) {
        return;
    }
    int one_bat_dim = comb_nodeNum * mixnodes;
    int my_batid = my_idx / one_bat_dim;
    int my_comb_col = (my_idx % one_bat_dim) / mixnodes;
    int my_mix_id = threadIdx.x;
    tmpPtr[my_mix_id] = mix_surface[my_mix_id * comb_nodeNum + my_comb_col] * raw_yFeaMat[my_batid * raw_nodeNum + comb_nodeNum + my_mix_id];
    __syncthreads();
    int thdIdx = threadIdx.x;
    int idx, incr, pos;
    for (idx = mixnodes; idx > 1; idx = incr) {
        incr = idx / 2;
        if (idx % 2 != 0) {
            ++incr;
        }
        if (thdIdx < incr) {
            pos = thdIdx + incr;
            if (pos < idx) {
                tmpPtr[thdIdx] += tmpPtr[pos];
            }
        }
        __syncthreads();
    }
    if (thdIdx == 0) {
        comb_yFeaMat[blockIdx.x] = /*raw_yFeaMat[my_batid * raw_nodeNum + comb_nodeNum + mixnodes] * tmpPtr[0] +*/ mixscaler * raw_yFeaMat[my_comb_col + my_batid * raw_nodeNum];
        //comb_yFeaMat[blockIdx.x] = tmpPtr[0] + mixscaler * raw_yFeaMat[my_comb_col + my_batid * raw_nodeNum];
        //comb_yFeaMat[blockIdx.x] = mixscaler * raw_yFeaMat[my_comb_col + my_batid * raw_nodeNum];
        /*if (comb_yFeaMat[blockIdx.x] < 0.0) {
            comb_yFeaMat[blockIdx.x] = 0.0;
        }
        if (comb_yFeaMat[blockIdx.x] > 1.0) {
            comb_yFeaMat[blockIdx.x] = 1.0;
        }
        */
        //if (comb_yFeaMat[blockIdx.x] < COMBLITTLE) {
        //    comb_yFeaMat[blockIdx.x] = COMBLITTLE;
        //}

    }
}

__global__ void HKern_CalcSoftmaxSum(NFloat *comb_softmaxSum, NFloat *raw_yFeaMat, int batLen, int raw_nodeNum, int comb_nodeNum, int mixnodes) {
    extern __shared__ NFloat tmpPtr[];
    int my_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (my_idx >= batLen * mixnodes) {
        return;
    }
    int thdIdx = threadIdx.x;
    int my_batid = blockIdx.x;
    tmpPtr[thdIdx] = raw_yFeaMat[my_batid * raw_nodeNum + comb_nodeNum + thdIdx];
    __syncthreads();
    int idx, incr, pos;
    for (idx = mixnodes; idx > 1; idx = incr) {
        incr = idx / 2;
        if (idx % 2 != 0) {
            ++incr;
        }
        if (thdIdx < incr) {
            pos = thdIdx + incr;
            if (pos < idx) {
                tmpPtr[thdIdx] += tmpPtr[pos];
            }
        }
        __syncthreads();
    }
    raw_yFeaMat[my_batid * raw_nodeNum + comb_nodeNum + thdIdx] /= tmpPtr[0];
    if (thdIdx == 0) {
        comb_softmaxSum[my_batid] = tmpPtr[0];
    }
}

__global__ void HKern_CalcSoftmaxMin(NFloat *comb_softmaxSum, NFloat *raw_yFeaMat, int batLen, int raw_nodeNum, int comb_nodeNum, int mixnodes) {
    extern __shared__ NFloat tmpPtr[];
    int my_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (my_idx >= batLen * mixnodes) {
        return;
    }
    int thdIdx = threadIdx.x;
    int my_batid = blockIdx.x;
    tmpPtr[thdIdx] = raw_yFeaMat[my_batid * raw_nodeNum + comb_nodeNum + thdIdx];
    __syncthreads();
    int idx, incr, pos;
    for (idx = mixnodes; idx > 1; idx = incr) {
        incr = idx / 2;
        if (idx % 2 != 0) {
            ++incr;
        }
        if (thdIdx < incr) {
            pos = thdIdx + incr;
            if (pos < idx) {
                if (tmpPtr[thdIdx] < tmpPtr[pos]) {
                    tmpPtr[thdIdx] = tmpPtr[pos];
                }
            }
        }
        __syncthreads();
    }
    raw_yFeaMat[my_batid * raw_nodeNum + comb_nodeNum + thdIdx] -= tmpPtr[0];
    
    NFloat tmpVal = raw_yFeaMat[my_batid * raw_nodeNum + comb_nodeNum + thdIdx];
    CHKNFLTEXPE(tmpVal)
    raw_yFeaMat[my_batid * raw_nodeNum + comb_nodeNum + thdIdx] = exp(tmpVal);
}


void FwdCombDNNandMixCUDA(NFloat *comb_yFeaMat, NFloat *mix_surface, NFloat *raw_yFeaMat, NFloat *comb_softmaxSum, int batLen, int raw_nodeNum, int comb_nodeNum, int mixnodes, float mixscaler) {
    int blks = comb_nodeNum * batLen;
    int sBytes = mixnodes * sizeof(NFloat);
    HKern_CalcSoftmaxMin<<< batLen, mixnodes, sBytes >>>(comb_softmaxSum, raw_yFeaMat, batLen, raw_nodeNum, comb_nodeNum, mixnodes);
    HKern_CalcSoftmaxSum<<< batLen, mixnodes, sBytes >>>(comb_softmaxSum, raw_yFeaMat, batLen, raw_nodeNum, comb_nodeNum, mixnodes);
    HKern_FwdCombDNNandMix<<< blks, mixnodes, sBytes >>>(comb_yFeaMat, mix_surface, raw_yFeaMat, batLen, raw_nodeNum, comb_nodeNum, mixnodes, mixscaler);
}

__global__ void HKern_ApplyActLHUCPenalty(NFloat *actscaler_elems, NFloat * dactscaler_elems, int nodeNum, float lhuc_penalty, int total_jobs, int grid_dim) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_jobs) {
        return;
    }
    int my_row = idx / grid_dim; 
    int my_col = idx % grid_dim;
    float tmpVal = 0;
    float myVal = actscaler_elems[idx];
    if (my_row - 1 >= 0) {
        tmpVal += myVal - actscaler_elems[(my_row - 1) * grid_dim + my_col];
    }
    if (my_row + 1 < grid_dim) {
        tmpVal += myVal - actscaler_elems[(my_row + 1) * grid_dim + my_col];
    }
    if (my_col - 1 >= 0) {
        tmpVal += myVal - actscaler_elems[idx - 1];
    }
    if (my_col + 1 < grid_dim) {
        tmpVal += myVal - actscaler_elems[idx + 1];
    }
    dactscaler_elems[idx] += lhuc_penalty * tmpVal;
}

void ApplyActLHUCPenaltyCUDA(NFloat * actscaler_elems, NFloat * dactscaler_elems, int nodeNum, float lhuc_penalty) {
    int total_jobs = nodeNum;
    int nBlocks = CEIL(total_jobs, THREADPERBLOCK);
    int grid_dim = (int)sqrt(nodeNum);
    HKern_ApplyActLHUCPenalty<<<nBlocks, THREADPERBLOCK>>>(actscaler_elems, dactscaler_elems, nodeNum, lhuc_penalty, total_jobs, grid_dim);
}





__global__ void HKern_ApplyActLHUCPenaltySoftCUDA(NFloat * d_lhuc, NFloat * lhuc, NFloat * phoneidx, NFloat * pos_dist, float lhuc_range_var, float lhuc_dist_var, int nodeNum, int batLen, NFloat * penal_val) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx = threadIdx.x; 
    int blkIdx = blockIdx.x;
    int thdNum = min(blockDim.x, nodeNum);
    
    int idx, base, pos, pos_dist_base, my_node_id, incr;

    float regVal = 0;
    float tmpVal, my_pos_dist, idx_pos_dist, my_lhuc, imp2, deltaVal;
    if (thdIdx < thdNum && blkIdx < batLen * nodeNum) {
        pos_dist_base = ((int)phoneidx[blkIdx / nodeNum]) * nodeNum;
        my_node_id = blkIdx % nodeNum;
        idx = thdIdx;
        tmpPtr[thdIdx] = 0;
        my_pos_dist = pos_dist[pos_dist_base + my_node_id];
        imp2 = exp(-my_pos_dist * my_pos_dist / lhuc_dist_var);
        my_lhuc = lhuc[my_node_id];
        while (idx < nodeNum) {
            idx_pos_dist = pos_dist[pos_dist_base + idx];
            tmpVal = idx_pos_dist - my_pos_dist;
            tmpVal = exp(-tmpVal * tmpVal / lhuc_range_var);
            regVal += tmpVal * (my_lhuc - lhuc[idx]) * (my_lhuc - lhuc[idx]) * imp2;
            deltaVal = tmpVal * (my_lhuc - lhuc[idx]) * imp2 + 
                    exp(-idx_pos_dist * idx_pos_dist / lhuc_dist_var) * tmpVal * (my_lhuc - lhuc[idx]);
            tmpPtr[thdIdx] += deltaVal;
            idx += thdNum;
        }
        atomicAdd(penal_val, regVal);
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        
        if (thdIdx == 0) {
            d_lhuc[blkIdx] = 2 * tmpPtr[0]; // * exp(-my_pos_dist * my_pos_dist / lhuc_dist_var);
        }
    }
}


__global__ void HKern_ApplyActLHUCPenaltySoftAddCUDA(NFloat * d_lhuc, NFloat * dest_d_lhuc, int nodeNum, int batLen, float lhuc_penalty) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx = threadIdx.x;
    int blkIdx = blockIdx.x;
    int thdNum = min(blockDim.x, batLen);
    int idx, pos, my_node_id, incr, step, tot;
    if (thdIdx < thdNum && blkIdx < nodeNum) {
        
        idx = thdIdx;
        step = thdNum * nodeNum;
        tmpPtr[thdIdx] = 0;
        tot = nodeNum * batLen;
        while (idx < tot) {
            tmpPtr[thdIdx] += d_lhuc[idx];
            idx += step;
        }
        __syncthreads();
        
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++ incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        if (thdIdx == 0) {
            /*
            int j;
            for (j = 0; j < thdNum; j += 1) {
                tmpPtr[0] += d_lhuc[j];
            }
            */
            dest_d_lhuc[blkIdx] += lhuc_penalty * tmpPtr[0];
        }
    }
}

void ApplyActLHUCPenaltySoftCUDA(NFloat * d_lhuc, NFloat * lhuc, NFloat * dest_d_lhuc, NFloat * phoneidx, NFloat * pos_dist, float lhuc_range_var, float lhuc_dist_var, int nodeNum, int batLen, float lhuc_penalty, NFloat * penal_val) {
    int nBlocks = batLen * nodeNum;
    int sBytes = THREADPERBLOCK * sizeof(NFloat);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyActLHUCPenaltySoftCUDA: Block number exceeds the maximum");
    HKern_ApplyActLHUCPenaltySoftCUDA<<<nBlocks, THREADPERBLOCK, sBytes>>>(d_lhuc, lhuc, phoneidx, pos_dist, lhuc_range_var, lhuc_dist_var, nodeNum, batLen, penal_val);
    //HKern_ApplyActLHUCPenaltyL2CUDA<<<nBlocks, THREADPERBLOCK, sBytes>>>(d_lhuc, lhuc, phoneidx, pos_dist, lhuc_range_var, lhuc_dist_var, nodeNum, batLen, penal_val);
    nBlocks = nodeNum;
    HKern_ApplyActLHUCPenaltySoftAddCUDA<<<nBlocks, THREADPERBLOCK, sBytes>>>(d_lhuc, dest_d_lhuc, nodeNum, batLen, lhuc_penalty);
}


__global__ void HKern_ApplyActLHUCPenaltyLocalSoftCUDA(NFloat * lhuc, NFloat * dest_d_lhuc, NFloat * d_lhuc_surface, int nodeNum, int batLen, float lhuc_penalty, NFloat * penalty_val) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx = threadIdx.x; 
    int blkIdx = blockIdx.x;
    int thdNum = min(blockDim.x, nodeNum);
    int idx, base, pos, incr;

    float regVal = 0;
    float tmpVal, my_lhuc;
    if (thdIdx < thdNum && blkIdx < nodeNum) {
        my_lhuc = lhuc[blkIdx];
        idx = thdIdx;
        tmpPtr[thdIdx] = 0;
        d_lhuc_surface += blkIdx * nodeNum + idx;
        while (idx < nodeNum) {
            tmpVal = my_lhuc - lhuc[idx];
            tmpPtr[thdIdx] += tmpVal * (*d_lhuc_surface);
            regVal += tmpVal * tmpVal * (*d_lhuc_surface);
            idx += thdNum;
            d_lhuc_surface += thdNum;
        }
        atomicAdd(penalty_val, 4 * regVal);
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        if (thdIdx == 0) {
            dest_d_lhuc[blkIdx] += lhuc_penalty * tmpPtr[0];
        }
    }
}
void ApplyActLHUCPenaltyLocalSoftCUDA(NFloat * lhuc, NFloat * dest_d_lhuc, NFloat * d_lhuc_surface, int nodeNum, int batLen, float lhuc_penalty, NFloat * penalty_val) {
    int nBlocks = nodeNum;
    int sBytes = THREADPERBLOCK * sizeof(NFloat);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyActLHUCPenaltyLocalSoftCUDA: Block number exceeds the maximum");
    //printf("ksksksk=%f\n", lhuc_penalty);exit(0);
    HKern_ApplyActLHUCPenaltyLocalSoftCUDA<<<nBlocks, THREADPERBLOCK, sBytes>>>(lhuc, dest_d_lhuc, d_lhuc_surface, nodeNum, batLen, lhuc_penalty, penalty_val);
}


__global__ void HKern_StimuGrad(NFloat * srcPtr, int row, int col, NFloat * grid_suface, int num_phone, NFloat * phone_vec, NFloat *dstPtr, NFloat * weight_norms) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, rowIdx, thdNum, base, idx, incr, pos;
    NFloat sumVal, tmpVal;
    NFloat a;
    

    thdIdx = threadIdx.x;   /* num threads per block */
    rowIdx = blockIdx.x;    /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, col);


    if (thdIdx < thdNum && rowIdx < row) {
        /* 1. find the sum Z */
        /* a. collect the sum for the groups */
        base = rowIdx * col;

        idx = thdIdx;
        tmpPtr[thdIdx] = 0.0;
        while (idx < col) {
            pos = base + idx;
            dstPtr[pos] = srcPtr[pos];
            if (dstPtr[pos] < LITTLE) {
                dstPtr[pos] = dstPtr[pos];
            }
            dstPtr[pos] *= weight_norms[idx];
            tmpPtr[thdIdx] += dstPtr[pos];
            idx += thdNum;
        }
        __syncthreads();
        /* b. dual add within current block */
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }

        sumVal = tmpPtr[0];

        /* 3. normalise */
        idx = thdIdx; 
        while (idx < col) {
            //if (abs(dstPtr[base + idx]) < LITTLE) {
            //    dstPtr[base + idx] = LITTLE;
            //}
            a = (1 / sumVal - grid_suface[(idx) * num_phone + (int)(phone_vec[rowIdx])] / dstPtr[base + idx]) * weight_norms[idx];
            dstPtr[base + idx] = a;
            idx += thdNum;
        }
    }
}

void StimuGradCUDA(NFloat * srcPtr, int row, int col, NFloat * grid_suface, int num_phone, NFloat * phone_vec, NFloat *dstPtr, NFloat * weight_norms) {
    int nBlocks, sBytes;

    //HError(9999, "row=%d col=%d num_phone=%d tpb=%d\n", row, col, num_phone, THREADPERBLOCK);

    nBlocks = row;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "StimuGradCUDA: Block number exceeds the maximum");
    HKern_StimuGrad<<<nBlocks, THREADPERBLOCK, sBytes>>>(srcPtr, row, col, grid_suface, num_phone, phone_vec, dstPtr, weight_norms);
}


__global__ void HKern_CalcWeightNorms(NFloat * weight_norms, NFloat * weights, int input_dim, int output_dim) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, rowIdx, thdNum, base, idx, incr, pos, step, total_weight;
    NFloat sumVal;


    thdIdx = threadIdx.x;
    rowIdx = blockIdx.x;
    thdNum = min(blockDim.x, output_dim);

    step = thdNum * input_dim;
    total_weight = input_dim * output_dim;

    if (thdIdx < thdNum && rowIdx < input_dim) {
        /*if (thdIdx == 0) {
            int xx = 0;
            weight_norms[rowIdx] = 0;
            for (xx = 0; xx < output_dim; ++ xx) {
                weight_norms[rowIdx] += weights[xx * input_dim + rowIdx] * weights[xx * input_dim + rowIdx];
            }
        }
        */
        
        base = thdIdx * input_dim + rowIdx;
        tmpPtr[thdIdx] = weights[base] * weights[base];
        base += step;
        while (base < total_weight) {
            tmpPtr[thdIdx] += weights[base] * weights[base];
            base += step;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        if (thdIdx == 0) {
            weight_norms[rowIdx] = sqrt(sumVal);
        }
    }
}

void CalcWeightNormsCUDA(NFloat * weight_norms, NFloat * weights, int input_dim, int output_dim) {
    int nBlocks, sBytes;
    nBlocks = input_dim;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "CalcWeightNormsCUDA: Block number exceeds the maximum");
    HKern_CalcWeightNorms<<<nBlocks, THREADPERBLOCK, sBytes>>>(weight_norms, weights, input_dim, output_dim);
}

__global__ void HKern_ApplyActStimuPenalty(NFloat * stimu_dy, NFloat * y, NFloat * actscaler, NFloat * dactscaler, int batLen, int nodeNum, float stimu_penalty) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, rowIdx, thdNum, base, idx, incr, pos, step, total_weight;
    NFloat sumVal;
    thdIdx = threadIdx.x;
    rowIdx = blockIdx.x;
    thdNum = min(blockDim.x, batLen);

    step = thdNum * nodeNum;
    total_weight = batLen * nodeNum;
    if (thdIdx < thdNum && rowIdx < nodeNum) {
        base = thdIdx * nodeNum + rowIdx;
        tmpPtr[thdIdx] = stimu_dy[base] * y[base];
        base += step;
        while (base < total_weight) {
            tmpPtr[thdIdx] += stimu_dy[base] * y[base];
            base += step;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        if (thdIdx == 0) {
            dactscaler[rowIdx] += stimu_penalty * sumVal / actscaler[rowIdx];
            //dactscaler[rowIdx] = -100;
        }
    }
}

void ApplyActStimuPenaltyCUDA(NFloat * stimu_dy, NFloat * y, NFloat * actscaler, NFloat * dactscaler, int batLen, int nodeNum, float stimu_penalty) {
    int nBlocks = nodeNum;
    int sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyActStimuPenaltyCUDA: Block number exceeds the maximum");
    HKern_ApplyActStimuPenalty<<<nBlocks, THREADPERBLOCK, sBytes>>>(stimu_dy, y, actscaler, dactscaler, batLen, nodeNum, stimu_penalty);
}

__global__ void HKern_CalcSumActi(NFloat * sum_xFeaMat, NFloat * xFeaMats, NFloat * weight_norms, int batLen, int nodeNum) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, rowIdx, thdNum, base, idx, incr, pos;
    NFloat sumVal, tmpVal;

    thdIdx = threadIdx.x;
    rowIdx = blockIdx.x;
    thdNum = min(blockDim.x, nodeNum);

    if (thdIdx < thdNum && rowIdx < batLen) {
        base = rowIdx * nodeNum;
        idx = thdIdx;
        /*if (idx < nodeNum) {
            tmpVal = xFeaMats[base + idx];
            if (tmpVal < LITTLE) tmpVal = LITTLE;
            tmpPtr[thdIdx] = tmpVal *  weight_norms[idx];
            idx += thdNum;
            if (idx < nodeNum) {
                tmpVal = xFeaMats[base + idx];
                tmpPtr[thdIdx] = tmpVal *  weight_norms[idx];
            }
        }
        */
        tmpPtr[thdIdx] = 0;
        while (idx < nodeNum) {
            tmpVal = xFeaMats[base + idx] * weight_norms[idx];
            if (tmpVal < LITTLE) tmpVal = LITTLE;
            tmpPtr[thdIdx] += tmpVal;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        
        if (thdIdx == 0) {
            sum_xFeaMat[rowIdx] = sumVal;
        }
        
        /*if (thdIdx == 0) {
            int j;
            sumVal = 0;
            for (j = 0; j < nodeNum; ++ j) {
               sumVal += xFeaMats[base + j] * weight_norms[j];
            }
            sum_xFeaMat[rowIdx] = sumVal;
        }
        */
    }
}

void CalcSumActiCUDA(NFloat * sum_xFeaMat_elems, NFloat * xFeaMats, NFloat * weight_norms, int batLen, int nodeNum) {
    int nBlocks, sBytes;
    nBlocks = batLen;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "CalcSumActiCUDA: Block number exceeds the maximum");
    HKern_CalcSumActi<<<nBlocks, THREADPERBLOCK, sBytes>>>(sum_xFeaMat_elems, xFeaMats, weight_norms, batLen, nodeNum);
}

__global__ void HKern_CalcStimuKL(NFloat *klvec, NFloat *yFeaMats, NFloat *weight_norms, NFloat *sum_yFeaMats, NFloat *acti_surface, NFloat *phoneidx_vec, int batLen, int nodeNum, int num_phone) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, thdNum, idx, incr, pos;
    NFloat sumVal, surfVal, tmpVal;

    thdIdx = threadIdx.x;
    int total = batLen * nodeNum;
    thdNum = min(blockDim.x, total);



    //int num_phone = 46; //TODO
    int phone_id, batid, nodeid;

    
    


    if (thdIdx < thdNum) {
        
        idx = thdIdx;

        
        tmpPtr[thdIdx] = 0;
        //tmpPtr[thdIdx] = surfVal * log(surfVal / (surfVal + 0.1));

        while (idx < total) {
            batid = idx / nodeNum;
            nodeid = idx % nodeNum;
            phone_id = (int)(phoneidx_vec[batid]);
            surfVal = acti_surface[phone_id + num_phone * nodeid];
            tmpVal = yFeaMats[idx];
            if (tmpVal < LITTLE) {
                tmpVal = LITTLE;
            }
            tmpPtr[thdIdx] += surfVal * log(surfVal * sum_yFeaMats[batid] / tmpVal / weight_norms[nodeid]);
            //tmpPtr[thdIdx] += surfVal * log(surfVal / (surfVal + 0.1));
            
            idx += thdNum;
            
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        sumVal = tmpPtr[0];
        if (thdIdx == 0) {
            klvec[0] = sumVal;
        }
    }
}

void CalcStimuKLCUDA(NFloat *klvec, NFloat *yFeaMats, NFloat *weight_norms, NFloat *sum_yFeaMats, NFloat *acti_surface, NFloat *phoneidx_vec, int batLen, int nodeNum, int num_phone) {
    int sBytes = sizeof(NFloat) * THREADPERBLOCK;
    HKern_CalcStimuKL<<<1, THREADPERBLOCK, sBytes>>>(klvec, yFeaMats, weight_norms, sum_yFeaMats, acti_surface, phoneidx_vec, batLen, nodeNum, num_phone);
}



//cw564 - stimu -- end


__global__ void HKern_ApplyRedSoftmaxAct(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, rowIdx, thdNum, base, idx, incr, pos;
    NFloat maxVal, sumVal, tmpVal;

    thdIdx = threadIdx.x;	/* num threads per block */
    rowIdx = blockIdx.x;	/* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, col);
    if (thdIdx < thdNum && rowIdx < row) {
        base = rowIdx * col;
        /* 1. find the max val for current frame (rowIdx) and store it in tmpPtr[thdIdx] */
        /* a. collect the maxes for the groups */
        idx = thdIdx;
        tmpPtr[thdIdx] = srcPtr[base + idx];
        idx += thdNum;
        while (idx < col) {
            pos = base + idx;
            if (tmpPtr[thdIdx] < srcPtr[pos])
                tmpPtr[thdIdx] = srcPtr[pos];
            idx += thdNum;
        }
        __syncthreads();
        /* b. dual max within current block */
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx && tmpPtr[thdIdx] < tmpPtr[pos]) {
                    tmpPtr[thdIdx] = tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        maxVal = tmpPtr[0];
        __syncthreads();
        /* 2. find the sum */
        /* a. collect the sum for the groups */
        idx = thdIdx;
        tmpPtr[thdIdx] = 0.0;
        while (idx < col) {
            pos = base + idx;
            tmpVal = srcPtr[pos] - maxVal;
            CHKNFLTEXPE(tmpVal)
            dstPtr[pos] = exp(tmpVal);
            tmpPtr[thdIdx] += dstPtr[pos];
            idx += thdNum;
        }
        __syncthreads();
        /* b. dual add within current block */
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }

        sumVal = tmpPtr[0];
        /* 3. normalise */
        idx = thdIdx; 
        while (idx < col) {
            dstPtr[base + idx] /= sumVal;
            idx += thdNum;
        }
    } 
}

__global__ void HKern_ApplySoftReLAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = srcPtr[pos];
        CHKNFLTEXPE(floatVal)
        dstPtr[pos] = log(1.0 + exp(floatVal));
    } 
}

__global__ void HKern_ApplyDSoftReLAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = srcPtr[pos];
        CHKNFLTEXPE(floatVal)
        dstPtr[pos] = 1.0 - 1.0 / exp(floatVal);
    }
}

__global__ void HKern_ApplySoftSignAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        dstPtr[pos] = srcPtr[pos] / (1 + abs(srcPtr[pos]));
    }
}

__global__ void HKern_ApplyLogTrans(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = srcPtr[pos];
        if (floatVal <= 0) {
            floatVal = LZERO;
        }
        else {        
            floatVal = log(floatVal);
            if (floatVal < LSMALL) {
                floatVal = LSMALL;
            }
        }
        dstPtr[pos] = floatVal;
    }
}

__global__ void HKern_RedSumNMatrixByColCUDA(NFloat *srcPtr, int row, int col, Boolean accFlag, NFloat *dstPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, colIdx, thdNum, base, idx, incr, pos, step;

    thdIdx = threadIdx.x;       /* num threads per block */
    colIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, row);
    if (thdIdx < thdNum && colIdx < col) {
        /*base = colIdx;*/
        /* collect the sums for the groups (and transpose the matrix) */
        tmpPtr[thdIdx] = 0.0;/*srcPtr[base + idx * col];*/
        base = colIdx;
        idx = thdIdx;
        pos = base + idx * col;
        step = thdNum * col;
        while (idx < row) {
            tmpPtr[thdIdx] += srcPtr[pos];
            pos += step;
            idx += thdNum;
        }
        __syncthreads();
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }

        /* copy the value to dstPtr */
        if (thdIdx == 0) {
            if (accFlag == FALSE) 
                dstPtr[colIdx] = 0.0; 
            dstPtr[colIdx] += tmpPtr[0];
        }
    }
}

__global__ void HKern_SumNMatrixByCol(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int i, pos;
    NFloat sum;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < col) {
        sum = 0.0;
        for (i = 0; i < row; ++i) {
            sum += srcPtr[i * col + pos];
        }
        dstPtr[pos] = sum;
    }
}

__global__ void HKern_SumNMatrixByColAcc(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int i, pos;
    NFloat sum;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < col) {
        sum = 0.0;
        for (i = 0; i < row; ++i) {
            sum += srcPtr[i * col + pos];
        }
        dstPtr[pos] += sum;
    }
}

__global__ void HKern_SquaredNSegment(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        dstPtr[pos] = pow(srcPtr[pos], 2);
    }
}

__global__ void HKern_CompAdaGradNSegment(NFloat eta, int K, int segLen, NFloat *ssgSeg, NFloat *nlrSeg) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        nlrSeg[pos] = eta / sqrt(K + ssgSeg[pos]);
    }
}

__global__ void HKern_CalXENTCriterionCUDA(NFloat *refPtr, NFloat *hypPtr, int segLen, NFloat *crtPtr) {
    __shared__ NFloat tmpPtr[THREADPERBLOCK];
    int thdIdx, thdNum, pos, idx, incr;
    NFloat tn, yn;

    thdIdx = threadIdx.x;
    thdNum = blockDim.x;

    if (thdIdx < thdNum) {
        /* a. collect the sums for the groups */
        pos = thdIdx;
        tmpPtr[thdIdx] = 0.0;
        while (pos < segLen) {
            tn = refPtr[pos];
            yn = hypPtr[pos];
            if (tn == 0.0) {
                tmpPtr[thdIdx] += 0.0;
            }
            else if (yn == 0.0) {
                tmpPtr[thdIdx] += tn * LZERO;
            }
            else {
                tmpPtr[thdIdx] += (-1.0) * tn * log(yn / tn); 
            }
            pos += thdNum;
        }
        __syncthreads();
        /* b. dual add within current block */
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        *crtPtr = tmpPtr[0];
    } 
}

__global__ void HKern_CalMMSECriterionCUDA(NFloat *refPtr, NFloat *hypPtr, int segLen, NFloat *crtPtr) {
    __shared__ NFloat tmpPtr[THREADPERBLOCK];
    int thdIdx, thdNum, pos, idx, incr;

    thdIdx = threadIdx.x;
    thdNum = blockDim.x;
    
    if (thdIdx < thdNum) {
        /* a. collect the sums for the groups */
        pos = thdIdx;
        tmpPtr[thdIdx] = 0.0;
        while (pos < segLen) {
            tmpPtr[thdIdx] += pow(refPtr[pos] - hypPtr[pos], 2);
            pos += thdNum;
        }
        __syncthreads();
        /* dual add within current block */
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        *crtPtr = tmpPtr[0];
    }
}

__global__ void HKern_AddSegmentTargetPen(NFloat *srcPtr, NFloat *penPtr, int row, int col, NFloat *dstPtr) {
    int pos, off;
    
    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < row * col) {
        off = pos % col;
        dstPtr[pos] = srcPtr[pos] + penPtr[off];
    }
}

/*__global__ void HKern_SubNSegmentByConst(NFloat *srcSeg, int segLen, float constVal, NFloat *dstSeg) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        dstSeg[pos] = srcSeg[pos] - constVal;
    }
}*/

/* cz277 - semi */
__global__ void HKern_ShiftNSegmentVals(NFloat *srcSeg, int segLen, float shiftVal, NFloat *dstSeg) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        dstSeg[pos] = srcSeg[pos] + shiftVal;
    }
}

/* cz277 - 1007 */
__global__ void HKern_CopyPartialNSegment(int minRow, int minCol, NFloat *srcPtr, int srcCol, NFloat *dstPtr, int dstCol) {
    int pos, rowIdx, colIdx;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < minRow * minCol) {
        rowIdx = pos / minCol;
        colIdx = pos % minCol;
        dstPtr[rowIdx * dstCol + colIdx] = srcPtr[rowIdx * srcCol + colIdx];
    }
}

/* cz277 - gradlim */
__global__ void HKern_ClipNSegmentVals(NFloat* srcSeg, int len, NFloat upperLim, NFloat lowerLim, NFloat *dstSeg) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        if (srcSeg[pos] > upperLim)
            dstSeg[pos] = upperLim;
        else if (srcSeg[pos] < lowerLim)
            dstSeg[pos] = lowerLim;
        else if (srcSeg != dstSeg)
            dstSeg[pos] = srcSeg[pos];
    }
}

__global__ void HKern_RedMaxElementIndex(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, rowIdx, thdNum, base, idx, incr, pos, off = THREADPERBLOCK;

    thdIdx = threadIdx.x;       /* num threads per block */
    rowIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, col);
    if (thdIdx < thdNum && rowIdx < row) {
        base = rowIdx * col;
        /* find the max val for current frame (rowIdx) and store it in tmpPtr[thdIdx] */
        /* a. collect the maxes for the groups */
        idx = thdIdx;
        tmpPtr[thdIdx] = srcPtr[base + idx];
        tmpPtr[off + thdIdx] = idx;
        idx += thdNum;
        while (idx < col) {
            pos = base + idx;
            if (tmpPtr[thdIdx] < srcPtr[pos]) {
                tmpPtr[thdIdx] = srcPtr[pos];
                tmpPtr[off + thdIdx] = idx;
            }
            idx += thdNum;
        }
        __syncthreads();
        /* b. dual max within current block */
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx && tmpPtr[thdIdx] < tmpPtr[pos]) {
                    tmpPtr[thdIdx] = tmpPtr[pos];
                    tmpPtr[off + thdIdx] = tmpPtr[off + pos];
                }
            }
            __syncthreads();
        }
        /*__syncthreads();*/
        if (thdIdx == 0)
            dstPtr[rowIdx] = tmpPtr[off + 0];
            /*dstPtr[rowIdx] = (NFloat) tmpPtr[off + 0];*/
        /*__syncthreads();*/
    }	
}

/* cz277 - max norm */
__global__ void HKern_RedCalExtNMatrixL2Norm(NFloat *matPtr, NFloat *vecPtr, int row, int col, NFloat *alphas) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, rowIdx, thdNum, base, idx, incr, pos;

    thdIdx = threadIdx.x;       /* num threads per block */
    rowIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, col);
    if (thdIdx < thdNum && rowIdx < row) {
        /* 1. accumulate the L2 norm for each row */
        base = rowIdx * col;
        idx = thdIdx;
        tmpPtr[thdIdx] = 0.0;
        while (idx < col) {
            pos = base + idx;
            tmpPtr[thdIdx] += pow(matPtr[pos], 2);
            idx += thdNum;
        }
        __syncthreads();
        /* b. dual add within current block */
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;                                   
	    }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx) {
                    tmpPtr[thdIdx] += tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        
        if (thdIdx == 0)
            alphas[rowIdx] = tmpPtr[0] + pow(vecPtr[rowIdx], 2);
    }
}

/* cz277 - max norm */
__global__ void HKern_RedMaxElementValue(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    extern __shared__ NFloat tmpPtr[];
    int thdIdx, rowIdx, thdNum, base, idx, incr, pos;

    thdIdx = threadIdx.x;       /* num threads per block */
    rowIdx = blockIdx.x;        /* block index, i.e., row num (minibatch size)  */
    thdNum = min(blockDim.x, col);
    if (thdIdx < thdNum && rowIdx < row) {
        base = rowIdx * col;
        /* find the max val for current frame (rowIdx) and store it in tmpPtr[thdIdx] */
        /* a. collect the maxes for the groups */
        idx = thdIdx;
        tmpPtr[thdIdx] = srcPtr[base + idx];
        idx += thdNum;
        while (idx < col) {
            pos = base + idx;
            if (tmpPtr[thdIdx] < srcPtr[pos]) {
                tmpPtr[thdIdx] = srcPtr[pos];
            }
            idx += thdNum;
        }
        __syncthreads();
        /* b. dual max within current block */
        for (idx = thdNum; idx > 1; idx = incr) {
            incr = idx / 2;
            if (idx % 2 != 0) {
                ++incr;
            }
            if (thdIdx < incr) {
                pos = thdIdx + incr;
                if (pos < idx && tmpPtr[thdIdx] < tmpPtr[pos]) {
                    tmpPtr[thdIdx] = tmpPtr[pos];
                }
            }
            __syncthreads();
        }
        /*__syncthreads();*/
        if (thdIdx == 0)
            dstPtr[rowIdx] = tmpPtr[0];
    }
}

/* --------------------------- HFBLat Kerns ------------------------ */

/* cz277 - cuda fblat */
__global__ void HKern_Setotprob4q(int T, NFloat *llhPtr, int ncols, int *qLo, int *qHi, int Q, float probScale, AcousticDev *acList) {
    int pos, tIdx, tRel, qIdx, s, Nq1;
    AcousticDev *curAc;
    NFloat *otprob;
    NFloat *matptr;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < T * Q) {
        tIdx = pos / Q + 1;
        qIdx = pos % Q + 1;
        if (qIdx >= qLo[tIdx] && qIdx <= qHi[tIdx]) {
            curAc = &acList[qIdx];
            Nq1 = curAc->Nq + 1;
            if (tIdx >= curAc->t_start && tIdx <= curAc->t_end) {	/* q is active at t */
                matptr = llhPtr + (tIdx - 1) * ncols;
                tRel = tIdx - curAc->t_start + 1;
                otprob = curAc->otprob + tRel * Nq1;
                for (s = 2; s < curAc->Nq; ++s) {
                    otprob[s] = matptr[curAc->indexes[s] - 1];
                }
            }
        }
    }
}


/* cz277 - cuda fblat */
__device__ NFloat LAddDev(NFloat x, NFloat y) {
    NFloat temp, diff, z;

    if (x < y) {
        temp = x;
        x = y;
        y = temp;
    }
    diff = y - x;
    if (diff < -23.025851) {
        if (x < LSMALL) {
            return LZERO;
        }
        else {
            return x;
        }
    }
    else {
        z = exp(diff);
        return x + log(1.0 + z);
    }
}

/* cz277 - cuda fblat */
__global__ void HKern_SetModelPlus(int Q, AcousticDev *acList) {
    int tIdx, tRel, qIdx, Nq1, i, j;
    AcousticDev *curAc;
    NFloat *bqt, *bqt1, x;

    qIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (qIdx < Q) {
        qIdx += 1;
        curAc = acList + qIdx;
        Nq1 = curAc->Nq + 1;
        for (tIdx = curAc->t_end; tIdx >= curAc->t_start; --tIdx) {
            tRel = tIdx - curAc->t_start + 1;
            /* SetModelPlus subroutine */
            x = LZERO;
            bqt = &curAc->betaPlus[tRel * Nq1];
            bqt1 = &curAc->betaPlus[(tRel + 1) * Nq1];
            if (tIdx == curAc->t_end) 
                bqt[curAc->Nq] = 0;
            else 
                bqt[curAc->Nq] = LZERO;
            for (i = 2; i < curAc->Nq; ++i) {
                x = bqt[curAc->Nq] + curAc->transp[i * Nq1 + curAc->Nq]; 
                if (tIdx + 1 <= curAc->t_end) {	/* in beam next time frame */
                    for (j = 2; j < curAc->Nq; ++j) {
                        x = LAddDev(x, bqt1[j] + curAc->transp[i * Nq1 + j]);
                    }
                }
                x += curAc->otprob[tRel * Nq1 + i];
                bqt[i] = x;
            }
            x = LZERO;
            for (i = 2; i < curAc->Nq; ++i) {
                x = LAddDev(x, bqt[i] + curAc->transp[1 * Nq1 + i]);
            }
            bqt[1] = x;
        }
        /* neet to set the total accumulated acoustics (tRel ~ tIdx = curAc->t_start) */
        if (curAc->SP == TRUE)
            curAc->aclike = curAc->transp[1 * Nq1 + curAc->Nq];
        else
            curAc->aclike = curAc->betaPlus[tRel * Nq1 + 1];
    }
}


/* cz277 - cuda fblat */
__global__ void HKern_ZeroAlphas(int T, int Q, AcousticDev *acList) {
    int i, pos, Nq1, tIdx, tRel, qIdx;
    AcousticDev *curAc;
    NFloat *alpha;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < T * Q) {
        tIdx = pos / Q + 1;
        qIdx = pos % Q + 1;
        curAc = &acList[qIdx];
        /* q is active at t */
        if (tIdx >= curAc->t_start && tIdx <= curAc->t_end) { 
            tRel = tIdx - curAc->t_start + 1;
            Nq1 = curAc->Nq + 1;
            alpha = &curAc->alphaPlus[tRel * Nq1];
            if (curAc->SP == FALSE) {
                for (i = 1; i < Nq1; ++i) {
                    alpha[i] = LZERO;    
                }
            }
        }
    }
}


/* cz277 - cuda fblat */
__global__ void HKern_StepAlpha(int Q, AcousticDev *acList) {
    int tIdx, qIdx, Nq1, i, j, tRel;
    AcousticDev *curAc;
    NFloat *aq, *laq, x = 0.0, y, a;
    NFloat *outprob;

    qIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (qIdx < Q) {
        qIdx += 1;
        curAc = acList + qIdx;
        /* for each time */
        for (tIdx = curAc->t_start; tIdx <= curAc->t_end; ++tIdx) {
            tRel = tIdx - curAc->t_start + 1;
            Nq1 = curAc->Nq + 1;
            aq = &curAc->alphaPlus[tRel * Nq1];
            laq = (tIdx - 1 >= curAc->t_start && tIdx - 1 <= curAc->t_end)? &curAc->alphaPlus[(tRel - 1) * Nq1]: NULL;
            /* outprob != NULL ?? */
            outprob = &curAc->otprob[tRel * Nq1];
            if (tIdx == curAc->t_start) 
                aq[1] = curAc->locc - curAc->aclike;
            else 
                aq[1] = LZERO;
            x = LZERO;
            for (j = 2; j < curAc->Nq; ++j) {
                a = curAc->transp[1 * Nq1 + j];
                x = (a > LSMALL)? a + aq[1]: LZERO;
                for (i = 2; i <= curAc->Nq; ++i) {
                    a = curAc->transp[i * Nq1 + j];
                    y = (laq? laq[i]: LZERO);
                    if (a > LSMALL && y > LSMALL) {
                        x = LAddDev(x, y + a);
                        /*x = log(x + y + a);*/
                    }
                }
                aq[j] = x + outprob[j];
            }
            x = LZERO;
            for (i = 2; i < curAc->Nq; ++i) {
                a = curAc->transp[i * Nq1 + curAc->Nq];
                y = aq[i];
                if (a > LSMALL && y > LSMALL) {
                    x = LAddDev(x, y + a);
                    /*x = log(x + y + a);*/
                }
            }
	    aq[curAc->Nq] = x;
            /* work out the exit problem for checking purpose */
        }
    }
}


/* --------------------------- Trace Flags ------------------------ */

/*  */
void SyncDev2Host(void *devPtr, void *hostPtr, size_t size) {
    cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
}

/*  */
void SyncHost2Dev(void *hostPtr, void *devPtr, size_t size) {
    cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);	
}

/*  */
void DevDispose(void *devPtr, size_t size) {
    cudaFree(devPtr);
    GPUMemUsed -= size;
}

/*  */
void DevNew(void **devAddr, size_t size) {
    cudaMalloc(devAddr, size);
    GPUMemUsed += size;
}

/*  */
void ShowGPUMemUsage(void) {
    printf("%dMB Memory Used on GPU %d\n", GPUMemUsed / 1048576, GPUDevId);
}

/*  */
void SetNSegmentCUDA(NFloat val, NFloat *segPtr, int segLen) {
    int nBlocks;

    nBlocks = CEIL(segLen, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "SetNSegmentCUDA: Block number exceeds the maximum");
    HKern_SetNSegment<<<nBlocks, THREADPERBLOCK>>>(val, segPtr, segLen);
}

/*  */
void ClearNSegmentCUDA(NFloat *segPtr, int segLen) {
    int nBlocks;
    cudaError_t status;

    nBlocks = CEIL(segLen, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ClearNSegmentCUDA: Block number exceeds the maximum");
    /*HKern_SetNSegment<<<nBlocks, THREADPERBLOCK>>>(0, segPtr, segLen);*/
    status = cudaMemset(segPtr, 0, segLen * sizeof(NFloat));
    if (status != cudaSuccess) {
        HError(9999, "ClearNSegmentCUDA: cudaMemset funtion failed");
    }
    /*cudaDeviceSynchronize();*/
}


/*  */
void CopyNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    cublasStatus_t status;

#ifdef DOUBLEANN
    status = cublasDcopy(handle, segLen, srcPtr, 1, dstPtr, 1);
#else
    status = cublasScopy(handle, segLen, srcPtr, 1, dstPtr, 1);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        HError(9999, "CopyNSegmentCUDA: CUBLAS library copy function failed");
    }
}

/*  */
void AddNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    cublasStatus_t status;
    const NFloat alpha = 1.0;

#ifdef DOUBLEANN
    status = cublasDaxpy(handle, segLen, &alpha, srcPtr, 1, dstPtr, 1);
#else
    status = cublasSaxpy(handle, segLen, &alpha, srcPtr, 1, dstPtr, 1);
#endif

    if (status != CUBLAS_STATUS_SUCCESS) {
        HError(9999, "AddNSegmentCUDA: CUBLAS library copy function failed");
    }
}

/* cz277 - l2 fix */
void AddScaledNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat scale, NFloat *dstPtr) {
    cublasStatus_t status;
    const NFloat alpha = scale;

#ifdef DOUBLEANN
    status = cublasDaxpy(handle, segLen, &alpha, srcPtr, 1, dstPtr, 1);
#else
    status = cublasSaxpy(handle, segLen, &alpha, srcPtr, 1, dstPtr, 1);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        HError(9999, "AddScaledNSegmentCUDA: CUBLAS library copy function failed");
    }
    
}

/*  */
void ScaleNSegmentCUDA(int segLen, NFloat scale, NFloat *valPtr) {
    cublasStatus_t status;

#ifdef DOUBLEANN
    status = cublasDscal(handle, segLen, &scale, valPtr, 1);
#else
    status = cublasSscal(handle, segLen, &scale, valPtr, 1);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        HError(9999, "ScaleNSegmentCUDA: CUBLAS library copy function failed");
    }
}

/*  */
void ScaledSelfAddNSegmentCUDA(NFloat *rhPtr, int segLen, NFloat scale, NFloat *lhPtr) {
    int nBlocks;
    
    nBlocks = CEIL(segLen, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ScaledSelfAddNSegmentCUDA: Block number exceeds the maximum");
    HKern_ScaledSelfAddNSegment<<<nBlocks, THREADPERBLOCK>>>(rhPtr, segLen, scale, lhPtr);
}

/*  */
void DupNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr, int times) {
    int nBlocks;

    nBlocks = CEIL(segLen * times, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "DupNSegmentCUDA: Block number exceeds the maximum");
    HKern_DupNSegment<<<nBlocks, THREADPERBLOCK>>>(srcPtr, segLen, dstPtr, times);
}

/*  */
void SubNSegmentCUDA(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    int nBlocks;
  
    nBlocks = CEIL(segLen, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "SubNSegmentCUDA: Block number exceeds the maximum");
    HKern_SubNSegment<<<nBlocks, THREADPERBLOCK>>>(lhPtr, rhPtr, segLen, resPtr);
}

/*  */
void MulNSegmentCUDA(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    int nBlocks;

    nBlocks = CEIL(segLen, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "MulNSegmentCUDA: Block number exceeds the maximum");
    HKern_MulNSegment<<<nBlocks, THREADPERBLOCK>>>(lhPtr, rhPtr, segLen, resPtr);
}

/* cz277 - pact */
void ApplyAffineActCUDA(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);    
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyAffineActCUDA: Block number exceeds the maximum");
    HKern_ApplyAffineAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, scalePtr, shiftPtr, dstPtr);
}

/* cz277 - pact */
void ApplyDAffineActCUDA(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDAffineActCUDA: Block number exceeds the maximum");
    HKern_ApplyDAffineAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, scalePtr, shiftPtr, dstPtr);
}


/* cz277 - pact */
void ApplyTrAffineActCUDA(NFloat *errPtr, NFloat *actPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, Boolean accFlag, NFloat *dScalePtr, NFloat *dShiftPtr) {
    int nBlocks, sBytes;

    nBlocks = col;
    sBytes = 2 * sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyTrStdDevAffineActCUDA: Block number exceeds the maximum");
    HKern_ApplyTrAffineAct<<<nBlocks, THREADPERBLOCK, sBytes>>>(errPtr, actPtr, row, col, scalePtr, shiftPtr, accFlag, dScalePtr, dShiftPtr);
}


/* cz277 - laf */
void AccMeanNSegmentCUDA(NFloat *valPtr, int row, int col, NFloat tSamp, NFloat *meanPtr) {
    int nBlocks, sBytes;
    
    nBlocks = col;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "AccMeanNSegmentCUDA: Block number exceeds the maximum");
    HKern_AccMeanNSegment<<<nBlocks, THREADPERBLOCK, sBytes>>>(valPtr, row, col, tSamp, meanPtr);
}

/* cz277 - laf */
void AccVarianceNSegmentCUDA(NFloat *valPtr, int row, int col, NFloat tSamp, NFloat *meanPtr, NFloat *varPtr) {
    int nBlocks, sBytes;
    
    nBlocks = col;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "AccVarianceNSegmentCUDA: Block number exceeds the maximum");
    HKern_AccVarianceNSegment<<<nBlocks, THREADPERBLOCK, sBytes>>>(valPtr, row, col, tSamp, meanPtr, varPtr);
}

/* cz277 - pact */
void ApplyParmReLUActCUDA(NFloat *srcPtr, int row, int col, NFloat *posPtr, NFloat *negPtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyParmReLUActCUDA: Block number exceeds the maximum");
    HKern_ApplyParmReLUAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, posPtr, negPtr, dstPtr);
}

/* cz277 - pact */
void ApplyDParmReLUActCUDA(NFloat *inpPtr, int row, int col, NFloat *posPtr, NFloat *negPtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDParmReLUActCUDA: Block number exceeds the maximum");
    HKern_ApplyDParmReLUAct<<<nBlocks, THREADPERBLOCK>>>(inpPtr, row, col, posPtr, negPtr, dstPtr);
}

/* cz277 - pact */
void ApplyTrParmReLUActCUDA(NFloat *errPtr, NFloat *inpPtr, int row, int col, Boolean accFlag, NFloat *dPosPtr, NFloat *dNegPtr) {
    int nBlocks, sBytes;

    nBlocks = col;
    sBytes = 2 * sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyTrParmReLUActCUDA: Block number exceeds the maximum");
    HKern_ApplyTrParmReLUAct<<<nBlocks, THREADPERBLOCK, sBytes>>>(errPtr, inpPtr, row, col, accFlag, dPosPtr, dNegPtr);
}

/* cz277 - pact */
void ApplyPReLUActCUDA(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyPReLUActCUDA: Block number exceeds the maximum");
    HKern_ApplyPReLUAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, scalePtr, dstPtr);
}

/* cz277 - pact */
void ApplyDPReLUActCUDA(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDPReLUActCUDA: Block number exceeds the maximum");
    HKern_ApplyDPReLUAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, scalePtr, dstPtr);
}

/* cz277 - pact */
void ApplyTrPReLUActCUDA(NFloat *errPtr, NFloat *srcPtr, int row, int col, NFloat *scalePtr, Boolean accFlag, NFloat *dScalePtr) {
    int nBlocks, sBytes;

    nBlocks = col;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyTrPReLUActCUDA: Block number exceeds the maximum");
    HKern_ApplyTrPReLUAct<<<nBlocks, THREADPERBLOCK, sBytes>>>(errPtr, srcPtr, row, col, scalePtr, accFlag, dScalePtr);
}

/*  */
void ApplyReLUActCUDA(NFloat *srcPtr, int len, NFloat scale, NFloat *dstPtr) {
    int nBlocks;
    
    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyReLActCUDA: Block number exceeds the maximum");
    HKern_ApplyReLUAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, scale, dstPtr);
}

/*  */
void ApplyDReLUActCUDA(NFloat *srcPtr, int len, NFloat scale, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDReLActCUDA: Block number exceeds the maximum");
    HKern_ApplyDReLUAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, scale, dstPtr);
}

/*  */
void ApplyDLinearActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDLinearActCUDA: Block number exceeds the maximum");
    HKern_ApplyDLinearAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

void ApplyLHUCSigmoidActCUDA(NFloat *srcPtr, int row, int col, NFloat *rolePtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyLHUCSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyLHUCSigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, rolePtr, dstPtr);
}

void ApplyDLHUCSigmoidActCUDA(NFloat *srcPtr, int row, int col, NFloat *rolePtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDLHUCSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyDLHUCSigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, rolePtr, dstPtr);
}

void ApplyTrLHUCSigmoidActCUDA(NFloat *errPtr, NFloat *actPtr, int row, int col, NFloat *rolePtr, Boolean accFlag, NFloat *dRolePtr) {
    int nBlocks, sBytes;

    nBlocks = col;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyTrLHUCSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyTrLHUCSigmoidActCUDA<<<nBlocks, THREADPERBLOCK, sBytes>>>(errPtr, actPtr, row, col, rolePtr, accFlag, dRolePtr); 
}

void ApplyPSigmoidActCUDA(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyPSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyPSigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, etaPtr, dstPtr);
}

void ApplyDPSigmoidActCUDA(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDPSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyDPSigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, etaPtr, dstPtr);
}

void ApplyTrPSigmoidActCUDA(NFloat *errPtr, NFloat *srcPtr, NFloat *etaPtr, int row, int col, Boolean accFlag, NFloat *dEtaPtr) {
    int nBlocks, sBytes;

    nBlocks = col;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyTrPSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyTrPSigmoidActCUDA<<<nBlocks, THREADPERBLOCK, sBytes>>>(errPtr, srcPtr, etaPtr, row, col, accFlag, dEtaPtr);
}


void ApplyParmSigmoidActCUDA(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat *thetaPtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyParmSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyParmSigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, etaPtr, gammaPtr, thetaPtr, dstPtr);
}

void ApplyDParmSigmoidActCUDA(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat *thetaPtr, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row * col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDParmSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyDParmSigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, etaPtr, gammaPtr, thetaPtr, dstPtr);
}

void ApplyTrParmSigmoidActCUDA(NFloat *errPtr, NFloat *inpPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat *thetaPtr, Boolean accFlag, NFloat *dEtaPtr, NFloat *dGammaPtr, NFloat *dThetaPtr) {
    int nBlocks, sBytes;

    nBlocks = col;
    sBytes = 3 * sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyTrParmSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyTrParmSigmoidActCUDA<<<nBlocks, THREADPERBLOCK, sBytes>>>(errPtr, inpPtr, row, col, etaPtr, gammaPtr, thetaPtr, accFlag, dEtaPtr, dGammaPtr, dThetaPtr);
}


/*  */
void ApplySigmoidActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplySigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplySigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

/*  */
void ApplyDSigmoidActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDSigmoidActCUDA: Block number exceeds the maximum");
    HKern_ApplyDSigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

/*  */
void ApplyTanHActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyTanHActCUDA: Block number exceeds the maximum");
    HKern_ApplyTanHAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

/*  */
void ApplyDTanHActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyDTanHActCUDA: Block number exceeds the maximum");
    HKern_ApplyDTanHAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}


/*  */
void ApplyRedSoftmaxActCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int nBlocks, sBytes;

    nBlocks = row;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyRedSoftmaxActCUDA: Block number exceeds the maximum");
    HKern_ApplyRedSoftmaxAct<<<nBlocks, THREADPERBLOCK, sBytes>>>(srcPtr, row, col, dstPtr);
}

/*  */
void ApplySoftmaxActCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplySoftmaxActCUDA: Block number exceeds the maximum");
    HKern_ApplySoftmaxAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, dstPtr);
}

/*  */
void ApplySoftReLActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;
 
    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplySoftReLActCUDA: Block number exceeds the maximum");
    HKern_ApplySoftReLAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

/*  */
void ApplyDSoftReLActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplySoftReLActCUDA: Block number exceeds the maximum");
    HKern_ApplyDSoftReLAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

/*  */
void ApplySoftSignActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplySoftSignActCUDA: Block number exceeds the maximum");
    HKern_ApplySoftSignAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);    
}

/*  */
void ApplyLogTransCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ApplyLogTransCUDA: Block number exceeds the maximum");
    HKern_ApplyLogTrans<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);    
}

/*  */
void RedSumNMatrixByColCUDA(NFloat *srcPtr, int row, int col, Boolean accFlag, NFloat *dstPtr) {
    int nBlocks, sBytes;

    nBlocks = col;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "RedSumNMatrixByColCUDA: Block number exceeds the maximum");
    HKern_RedSumNMatrixByColCUDA<<<nBlocks, THREADPERBLOCK, sBytes>>>(srcPtr, row, col, accFlag, dstPtr);
}

/*  */
void SumNMatrixByColCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(col, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "SumNMatrixByColCUDA: Block number exceeds the maximum");
    HKern_SumNMatrixByCol<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, dstPtr);
}

/*  */
void SquaredNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(segLen, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "SquaredNSegmentCUDA: Block number exceeds the maximum");
    HKern_SquaredNSegment<<<nBlocks, THREADPERBLOCK>>>(srcPtr, segLen, dstPtr);
}

/*  */
void CompAdaGradNSegmentCUDA(NFloat eta, int K, int segLen, NFloat *ssgSeg, NFloat *nlrSeg) {
    int nBlocks;

    nBlocks = CEIL(segLen, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "CompAdaGradNSegmentCUDA: Block number exceeds the maximum");
    HKern_CompAdaGradNSegment<<<nBlocks, THREADPERBLOCK>>>(eta, K, segLen, ssgSeg, nlrSeg);
}

/*  */
void HNBlasNNgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    cublasStatus_t status;

#ifdef DOUBLEANN
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
#else
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        HError(9999, "HNBlasNNgemmCUDA: CUBLAS library gemm function failed");
    }
}

/*  */
void HNBlasNTgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    cublasStatus_t status;

#ifdef DOUBLEANN
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, n, &beta, C, m);
#else
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, n, &beta, C, m);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        HError(9999, "HNBlasNTgemmCUDA: CUBLAS library gemm function failed");
    }
}

/*  */
void HNBlasTNgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    cublasStatus_t status;

#ifdef DOUBLEANN
    status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, k, B, k, &beta, C, m);
#else
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, k, B, k, &beta, C, m);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        HError(9999, "HNBlasTNgemmCUDA: CUBLAS library gemm function failed");
    }
}

/*  */
void CalXENTCriterionCUDA(NFloat *refPtr, NFloat *hypPtr, int segLen, NFloat *crtPtr) {
    HKern_CalXENTCriterionCUDA<<<1, THREADPERBLOCK>>>(refPtr, hypPtr, segLen, crtPtr);
}

/*  */
void CalMMSECriterionCUDA(NFloat *refPtr, NFloat *hypPtr, int segLen, NFloat *crtPtr) {
    HKern_CalMMSECriterionCUDA<<<1, THREADPERBLOCK>>>(refPtr, hypPtr, segLen, crtPtr);
}

/*  */
void AddNSegmentTargetPenCUDA(NFloat *srcSeg, NFloat *penSeg, int row, int col, NFloat *dstSeg) {
    int nBlocks, size;

    size = row * col;
    nBlocks = CEIL(size, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "AddNVectorTargetPenCUDA: Block number exceeds the maximum");

    HKern_AddSegmentTargetPen<<<nBlocks, THREADPERBLOCK>>>(srcSeg, penSeg, row, col, dstSeg);
}

void FindMaxElementCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int nBlocks, sBytes;

    nBlocks = row;
    sBytes = 2 * sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "FindMaxElementCUDA: Block number exceeds the maximum");
    HKern_RedMaxElementIndex<<<nBlocks, THREADPERBLOCK, sBytes>>>(srcPtr, row, col, dstPtr);
}

/*  */
/*void SubNSegmentByConstCUDA(NFloat *srcSeg, int segLen, NFloat constVal, NFloat *dstSeg) {
    int nBlocks;

    nBlocks = CEIL(segLen, THREADPERBLOCK); 
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "SubNSegmentByConstCUDA: Block number exceeds the maximum");

    HKern_SubNSegmentByConst<<<nBlocks, THREADPERBLOCK>>>(srcSeg, segLen, constVal, dstSeg);
}*/

/* cz277 - semi */
/*  */
void ShiftNSegmentValsCUDA(NFloat *srcSeg, int segLen, NFloat shiftVal, NFloat *dstSeg) {
    int nBlocks;

    nBlocks = CEIL(segLen, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ShiftNSegmentValsCUDA: Block number exceeds the maximum");

    HKern_ShiftNSegmentVals<<<nBlocks, THREADPERBLOCK>>>(srcSeg, segLen, shiftVal, dstSeg);
}

/* cz277 - 1007 */
void CopyPartialNSegmentCUDA(int minRow, int minCol, NFloat *srcPtr, int srcCol, NFloat *dstPtr, int dstCol) {
    int len, nBlocks;

    len = minRow * minCol;
    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "CopyPartialNSegmentCUDA: Block number exceeds the maximum");
    HKern_CopyPartialNSegment<<<nBlocks, THREADPERBLOCK>>>(minRow, minCol, srcPtr, srcCol, dstPtr, dstCol);
}

/* --------------------------- HFBLat funcs ------------------------ */

/* cz277 - cuda fblat */
void SetModelBetaPlusCUDA(int T, NMatrix *llhMat, int *qLo, int *qHi, int Q, float probScale, AcousticDev *acList) {
    int nBlocks;

    /* t in [1 ... T]; q in [1 ... Q] */
    nBlocks = CEIL(T * Q, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "SetModelBetaPlusCUDA: Block number exceeds the maximum");
    /* setotprob */
    HKern_Setotprob4q<<<nBlocks, THREADPERBLOCK>>>(T, llhMat->devElems, llhMat->colNum, qLo, qHi, Q, probScale, acList);
    /* set model beta plus */
    nBlocks = CEIL(Q, THREADPERBLOCK);
    HKern_SetModelPlus<<<nBlocks, THREADPERBLOCK>>>(Q, acList);

} 


/* cz277 - cuda fblat */
void ZeroAlphasCUDA(int T, int Q, AcousticDev *acList) {
    int nBlocks;

    nBlocks = CEIL(T * Q, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "ZeroAlphasCUDA: Block number exceeds the maximum");
    HKern_ZeroAlphas<<<nBlocks, THREADPERBLOCK>>>(T, Q, acList);
}


/* cz277 - cuda fblat */
void StepAlphaCUDA(int Q, AcousticDev *acList) {
    int nBlocks;

    nBlocks = CEIL(Q, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "StepAlphaCUDA: Block number exceeds the maximum");
    HKern_StepAlpha<<<nBlocks, THREADPERBLOCK>>>(Q, acList);
}

/* cz277 - gradlim */
void ClipNSegmentValsCUDA(NFloat* srcSeg, int len, NFloat upperLim, NFloat lowerLim, NFloat *dstSeg) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "LimitNSegmentValsCUDA: Block number exceeds the maximum");
    HKern_ClipNSegmentVals<<<nBlocks, THREADPERBLOCK>>>(srcSeg, len, upperLim, lowerLim, dstSeg);
}

/* cz277 - max norm */
void CalExtNMatrixL2NormCUDA(NFloat *matPtr, NFloat *vecPtr, int row, int col, NFloat *alphas) {
    int nBlocks, sBytes;
  
    nBlocks = row;
    sBytes = sizeof(NFloat) * THREADPERBLOCK;
    if (nBlocks > MAXBLOCKNUM)
        HError(9999, "CalExtNMatrixL2NormCUDA: Block number exceeds the maximum");
    HKern_RedCalExtNMatrixL2Norm<<<nBlocks, THREADPERBLOCK, sBytes>>>(matPtr, vecPtr, row, col, alphas);
    nBlocks = 1;
    HKern_RedMaxElementValue<<<nBlocks, THREADPERBLOCK, sBytes>>>(alphas, 1, row, alphas);
}


/* --------------------------- Trace Flags ------------------------ */


#ifdef __cplusplus
}
#endif


