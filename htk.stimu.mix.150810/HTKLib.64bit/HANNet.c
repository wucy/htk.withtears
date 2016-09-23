/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*      Machine Intelligence Laboratory                        */
/*      Cambridge University Engineering Department            */
/*      http://mil.eng.cam.ac.uk/                              */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http://www.microsoft.com                 */
/*                                                             */
/*              2002  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*         File: HANNet.c  ANN Model Definition Data Type      */
/* ----------------------------------------------------------- */

char *hannet_version = "!HVER!HANNet:   3.4.1 [CUED 30/11/13]";
char *hannet_vc_id = "$Id: HANNet.c,v 1.1.1.1 2013/11/13 09:54:58 cz277 Exp $";

#include "cfgs.h"
#include <time.h>
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HWave.h"
#include "HAudio.h"
#include "HParm.h"
#include "HLabel.h"
#include "HANNet.h"
#include "HModel.h"
#include "HTrain.h"
#include "HNet.h"
#include "HArc.h"
#include "HFBLat.h"
#include "HDict.h"
#include "HAdapt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/* ------------------------------ Trace Flags ------------------------------ */

static int trace = 0;

#define T_TOP 0001
#define T_CCH 0002

/* --------------------------- Memory Management --------------------------- */


/* ----------------------------- Configuration ------------------------------*/

static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;
static size_t batchSamples = 1;                 /* the number of samples in batch; 1 sample by default */
static char *updtFlagStr = NULL;                /* the string pointer indicating the layers to update */
static int updtIdx = 0;                         /* the index of current update*/
static Boolean hasShownUpdtFlag = FALSE;
/* cz277 - 1007 */
static int batIdx = 0;
/* cz277 - xform */
static RPLInfo *rplNMatInfo[MAXNUMRPLNMAT];
static RPLInfo *rplNVecInfo[MAXNUMRPLNVEC];

/* get the batch size */
int GetNBatchSamples(void) {
    return batchSamples;
}

/* set the batch size */
void SetNBatchSamples(int userBatchSamples) {
    batchSamples = userBatchSamples;
#ifdef CUDA
    RegisterTmpNMat(1, batchSamples);
#endif
}

/* set the index of current update */
void SetUpdateIndex(int curUpdtIdx) {
    updtIdx = curUpdtIdx;
}

/* get the index of current update */
int GetUpdateIndex(void) {
    return updtIdx;
}

/*  */
void SetBatchIndex(int curBatIdx) {
    batIdx = curBatIdx;
}

/*  */
int GetBatchIndex(void) {
    return batIdx;
}

/* cz277 - xform */
void InitRPLInfo(RPLInfo *rplInfo) {
    rplInfo->nSpkr = 0;
    rplInfo->inRPLMask = NULL;
    rplInfo->curOutSpkr = NewString(&gcheap, MAXSTRLEN);
    rplInfo->curInSpkr = NewString(&gcheap, MAXSTRLEN);
    rplInfo->cacheInSpkr = NewString(&gcheap, MAXSTRLEN);
    rplInfo->inRPLDir = NULL;
    rplInfo->inRPLExt = NULL;
    rplInfo->outRPLDir = NULL;
    rplInfo->outRPLExt = NULL;
    rplInfo->inActive = FALSE;
    rplInfo->outActive = FALSE;
    rplInfo->saveBinary = FALSE;
    rplInfo->rplNMat = NULL;
    memset(&rplInfo->saveRplNMatHost, 0, sizeof(NMatHost));
    /*rplInfo->saveRplNMat = NULL;*/
    rplInfo->rplNVec = NULL;
    memset(&rplInfo->saveRplNVecHost, 0, sizeof(NVecHost));
    /*rplInfo->saveRplNVec = NULL;*/
}

/*  */
void InitANNet(void)
{
    int intVal, i;
    char buf[MAXSTRLEN], cmd[MAXSTRLEN];

    Register(hannet_version, hannet_vc_id);
    nParm = GetConfig("HANNET", TRUE, cParm, MAXGLOBS);

    if (nParm > 0) {
        if (GetConfInt(cParm, nParm, "TRACE", &intVal)) { 
            trace = intVal;
        }
        if (GetConfInt(cParm, nParm, "BATCHSAMP", &intVal)) {
            if (intVal <= 0) {
                HError(9999, "InitANNet: Fail to set batch size");
            }
            SetNBatchSamples(intVal);
        }
        if (GetConfStr(cParm, nParm, "UPDATEFLAGS", buf)) {
            updtFlagStr = CopyString(&gcheap, buf);
        }
        /* cz277 - xform */
        for (i = 1; i < MAXNUMRPLNMAT; ++i) {
            rplNMatInfo[i] = NULL;
            /* mask */
            sprintf(cmd, "REPLACEABLENMATMASK%d", i);
            if (GetConfStr(cParm, nParm, cmd, buf)) {
                rplNMatInfo[i] = (RPLInfo *) New(&gcheap, sizeof(RPLInfo));
                InitRPLInfo(rplNMatInfo[i]);
                rplNMatInfo[i]->inRPLMask = CopyString(&gcheap, buf); 
            }
            if (rplNMatInfo[i] != NULL) {
                /* in dir */
                sprintf(cmd, "REPLACEABLENMATINDIR%d", i);
                if (GetConfStr(cParm, nParm, cmd, buf)) {
                    rplNMatInfo[i]->inRPLDir = CopyString(&gcheap, buf);
                }
                /* ext */
                sprintf(cmd, "REPLACEABLENMATEXT%d", i);
                if (GetConfStr(cParm, nParm, cmd, buf)) {
                    rplNMatInfo[i]->inRPLExt = CopyString(&gcheap, buf);
                    rplNMatInfo[i]->outRPLExt = CopyString(&gcheap, buf);
                }
                /* out dir */
                sprintf(cmd, "REPLACEABLENMATOUTDIR%d", i);
                if (GetConfStr(cParm, nParm, cmd, buf)) {
                    rplNMatInfo[i]->outRPLDir = CopyString(&gcheap, buf);
                }
            }
        }
        for (i = 1; i < MAXNUMRPLNVEC; ++i) {
            rplNVecInfo[i] = NULL;
            /* mask */
            sprintf(cmd, "REPLACEABLENVECMASK%d", i);
            if (GetConfStr(cParm, nParm, cmd, buf)) {
                rplNVecInfo[i] = (RPLInfo *) New(&gcheap, sizeof(RPLInfo));
                InitRPLInfo(rplNVecInfo[i]);
                rplNVecInfo[i]->inRPLMask = CopyString(&gcheap, buf);
            }
            if (rplNVecInfo[i] != NULL) {
                /* in dir */
                sprintf(cmd, "REPLACEABLENVECINDIR%d", i);
                if (GetConfStr(cParm, nParm, cmd, buf)) {
                    rplNVecInfo[i]->inRPLDir = CopyString(&gcheap, buf);
                }
                /* ext */
                sprintf(cmd, "REPLACEABLENVECEXT%d", i);
                if (GetConfStr(cParm, nParm, cmd, buf)) {
                    rplNVecInfo[i]->inRPLExt = CopyString(&gcheap, buf);
                    rplNVecInfo[i]->outRPLExt = CopyString(&gcheap, buf);
                }
                /* out dir */
                sprintf(cmd, "REPLACEABLENVECOUTDIR%d", i);
                if (GetConfStr(cParm, nParm, cmd, buf)) {
                    rplNVecInfo[i]->outRPLDir = CopyString(&gcheap, buf);
                }
            }
        }
    }

    if (TRUE) {
            /* GPU/MKL/CPU */              /* discard: should be set when compiling */
            /* THREADS */
            /* SGD/HF */
            /* LEARNING RATE SCHEDULE */
            /*     RELATED STUFFS */
    }
}

/* cz277 - xform */
Boolean LoadInRplNMat(ANNSet *annSet, NMatrix *rplNMat) {
    int rplIdx = rplNMat->rplIdx;

    if (rplIdx < 1 || rplIdx >= MAXNUMRPLNMAT) {
        HError(9999, "LoadInRplNMat: Replaceable matrix index %d out of range", rplIdx);
    }
    if (rplNMatInfo[rplIdx] == NULL) {
        /*HError(9999, "LoadInRplNMat: XForm info for replaceable matrix %d unset", rplIdx);*/
        return TRUE;
    }
    if (annSet->rplNMatInfo[rplIdx] == NULL) {
        annSet->rplNMatInfo[rplIdx] = rplNMatInfo[rplIdx];
    }
    else if (annSet->rplNMatInfo[rplIdx] != rplNMatInfo[rplIdx]) {
        HError(9999, "LoadInRplNMat: A different replaceable part info was loaded");
    }
    if (annSet->rplNMatInfo[rplIdx]->rplNMat == NULL) {
        annSet->rplNMatInfo[rplIdx]->rplNMat = rplNMat;
        annSet->rplNMatInfo[rplIdx]->rplNMat->rplIdx = rplIdx;
        /*annSet->rplNMatInfo[rplIdx]->saveRplNMat = CloneNMatrix(&gcheap, &rplNMat[rplIdx]);*/
        annSet->rplNMatInfo[rplIdx]->saveRplNMatHost.rowNum = rplNMat->rowNum;
        annSet->rplNMatInfo[rplIdx]->saveRplNMatHost.colNum = rplNMat->colNum;
        annSet->rplNMatInfo[rplIdx]->saveRplNMatHost.matElems = rplNMat->matElems;
        annSet->rplNMatInfo[rplIdx]->inActive = TRUE;
    }
    else if (annSet->rplNMatInfo[rplIdx]->rplNMat != rplNMat) {
        HError(9999, "LoadInRplNMat: A different NMatrix was set");
	/*annSet->rplNMatInfo[rplIdx]->rplNMat = rplNMat;*/
    }

    return TRUE;
}

/* cz277 - xform */ 
Boolean LoadInRplNVec(ANNSet *annSet, NVector *rplNVec) {
    int rplIdx = rplNVec->rplIdx;

    if (rplIdx < 1 || rplIdx >= MAXNUMRPLNVEC) {
        HError(9999, "LoadInRplNVec: Replaceable vector index %d out of range", rplIdx);
    }
    if (rplNVecInfo[rplIdx] == NULL) {
        /*HError(9999, "LoadInRplNVec: Mask for replaceable vector %d unset", rplIdx);*/
        return TRUE;
    }
    if (annSet->rplNVecInfo[rplIdx] == NULL) {
        annSet->rplNVecInfo[rplIdx] = rplNVecInfo[rplIdx];
    }
    else if (annSet->rplNVecInfo[rplIdx] != rplNVecInfo[rplIdx]) {
        HError(9999, "LoadInRplNVec: A different replaceable part info was loaded");
    }
    if (annSet->rplNVecInfo[rplIdx]->rplNVec == NULL) {
        annSet->rplNVecInfo[rplIdx]->rplNVec = rplNVec;
        annSet->rplNVecInfo[rplIdx]->rplNVec->rplIdx = rplIdx;
        /*annSet->rplNVecInfo[rplIdx]->saveRplNVec = CloneNVector(&gcheap, &rplNVec[rplIdx]);*/
        annSet->rplNVecInfo[rplIdx]->saveRplNVecHost.vecLen = rplNVec->vecLen;
        annSet->rplNVecInfo[rplIdx]->saveRplNVecHost.vecElems = rplNVec->vecElems;
        annSet->rplNVecInfo[rplIdx]->inActive = TRUE;
    }
    else if (annSet->rplNVecInfo[rplIdx]->rplNVec != rplNVec) {
        HError(9999, "LoadInRplNVec: A different NVector was set");
	/*annSet->rplNVecInfo[rplIdx]->rplNVec = rplNVec;*/
    }

    return TRUE;
}

/* set the update flag for each ANN layer */
void SetUpdateFlags(ANNSet *annSet) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    char *charPtr = NULL;
    char buf[MAXSTRLEN];
    
    if ((trace & T_TOP) && (hasShownUpdtFlag == FALSE)) {
        printf("SetUpdateFlags: Updating ");
    }

    if (updtFlagStr != NULL) {
        strcpy(buf, updtFlagStr);
        charPtr = strtok(buf, ",");
        /*charPtr = strtok(updtFlagStr, ",");*/
    }
    curAI = annSet->defsHead;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            if (charPtr != NULL) {
                layerElem->trainInfo->updtFlag = atoi(charPtr);
                charPtr = strtok(NULL, ",");
            }
            else {
                layerElem->trainInfo->updtFlag = ACTFUNUK | BIASUK | WEIGHTUK;
            }
            if ((trace & T_TOP) && (hasShownUpdtFlag == FALSE)) {
                if (!(layerElem->trainInfo->updtFlag & (ACTFUNUK | BIASUK | WEIGHTUK | INITUK))) {
                    printf(", NoParam");
                }
                else {
                    printf(", ");
                    if (layerElem->trainInfo->updtFlag & ACTFUNUK) { 
                        printf("+ActFun");
                    }
                    if (layerElem->trainInfo->updtFlag & BIASUK) { 
                        printf("+Bias");
                    }
                    if (layerElem->trainInfo->updtFlag & WEIGHTUK) {
                        printf("+Weight");
                    }
                    /* cz277 - pact */
                    if (layerElem->trainInfo->updtFlag & INITUK) {
                        printf("+ActFun(Init)");
                    }
                }
            }
        }
        curAI = curAI->next;
    }

    if ((trace & T_TOP) && (hasShownUpdtFlag == FALSE)) {
        printf("\n");
        hasShownUpdtFlag = TRUE;
    }
}

static inline void FillBatchFromFeaMixOLD(FeaMix *feaMix, int batLen, int *CMDVecPL) {
    int i, j, k, srcOff = 0, curOff = 0, dstOff, hisOff, hisDim;
    FELink feaElem;

    /* if it is the shared */
    if (feaMix->feaList[0]->feaMats[0] == feaMix->mixMats[0]) {
        return;
    }
    /* cz277 - 1007 */
    if (feaMix->batIdx > batIdx + 1 || feaMix->batIdx < batIdx) {
        HError(9999, "FillBatchFromFeaMix: batIdx of this feature mix does not match the global index");
    }
    else if (feaMix->batIdx == batIdx + 1) {
        return;
    }
    else {
        ++feaMix->batIdx;
    }

    /* otherwise, fill the batch with a mixture of the FeaElem */
    for (i = 0; i < feaMix->elemNum; ++i) {
        feaElem = feaMix->feaList[i];

        if (feaElem->inputKind == INPFEAIK || feaElem->inputKind == AUGFEAIK) {
            for (j = 0, srcOff = 0, dstOff = curOff; j < batLen; ++j, srcOff += feaElem->extDim, dstOff += feaMix->mixDim) {
                CopyNSegment(feaElem->feaMats[0], srcOff, feaElem->extDim, feaMix->mixMats[0], dstOff);
            }
        }
        else if (feaElem->inputKind == ANNFEAIK) {  /* ANNFEAIK, left context is consecutive */
            for (j = 0; j < batLen; ++j) {
                /* cz277 - gap */
                hisDim = feaElem->hisLen * feaElem->feaDim;
                hisOff = j * hisDim;
                if (CMDVecPL != NULL && feaElem->hisMat != NULL) {
                    if (CMDVecPL[j] == 0) {	/* reset the history */
			ClearNMatrixSegment(feaElem->hisMat, hisOff, hisDim);
                    }
                    else if (CMDVecPL[j] > 0) {	/* shift the history */
                        CopyNSegment(feaElem->hisMat, CMDVecPL[j] * hisDim, hisDim, feaElem->hisMat, hisOff);
                    }
                }
                /* standard operations */
                dstOff = j * feaMix->mixDim + curOff;
                for (k = 1; k <= feaElem->ctxMap[0]; ++k, dstOff += feaElem->feaDim) { 
                    if (feaElem->ctxMap[k] < 0) {
                        /* first, previous segments from hisMat to feaMix->mixMat */
                        srcOff = ((j + 1) * feaElem->hisLen + feaElem->ctxMap[k]) * feaElem->feaDim;
                        CopyNSegment(feaElem->hisMat, srcOff, feaElem->feaDim, feaMix->mixMats[0], dstOff);
                    }
                    else if (feaElem->ctxMap[k] == 0) {
                        /* second, copy current segment from feaMat to feaMix->mixMat */
                        srcOff = j * feaElem->srcDim + feaElem->dimOff;
                        CopyNSegment(feaElem->feaMats[0], srcOff, feaElem->feaDim, feaMix->mixMats[0], dstOff);
                    }
                    else {
                        HError(9999, "FillBatchFromFeaMix: The future of ANN features are not applicable");
                    }
                }
                /* shift history info in hisMat and copy current segment from feaMat to hisMat */
                if (feaElem->hisMat != NULL) {
                    dstOff = hisOff;
                    srcOff = dstOff + feaElem->feaDim;
                    for (k = 0; k < feaElem->hisLen - 1; ++k, srcOff += feaElem->feaDim, dstOff += feaElem->feaDim) {
                        CopyNSegment(feaElem->hisMat, srcOff, feaElem->feaDim, feaElem->hisMat, dstOff);    
                    }
                    srcOff = j * feaElem->srcDim + feaElem->dimOff;
                    CopyNSegment(feaElem->feaMats[0], srcOff, feaElem->feaDim, feaElem->hisMat, dstOff);
                }
            }
        }
        curOff += feaElem->extDim;
    }
}

/* cz277 - xform */
static inline void FillBatchFromFeaMix(LELink layerElem, int batLen) {
    int i, j, k, l, n, m, srcOff, curOff, dstOff, t, c, curCtx, pos;
    FELink feaElem;
    FeaMix *feaMix;
    NMatrix *mixMat, *feaMat;

    feaMix = layerElem->feaMix;
    /* if mixMats are shared with feaMats -- no need to reload the features */
    if (feaMix->elemNum == 1) {
        feaElem = feaMix->feaList[0];
        if (!(feaElem->inputKind == ANNFEAIK && IntVecSize(feaElem->ctxMap) > 1))
            return;
    }
    /* if feaMat is shared and is processed for current batch */
    if (feaMix->batIdx > batIdx + 1 || feaMix->batIdx < batIdx)
        HError(9999, "FillBatchFromFeaMix: batIdx of this feature mix does not match the global index");
    else if (feaMix->batIdx == batIdx + 1)
        return;
    else
        ++feaMix->batIdx;
    /* otherwise */
    n = IntVecSize(layerElem->drvCtx);
    for (i = 1, j = 1; i <= n; ++i) {
        while (layerElem->drvCtx[i] != feaMix->ctxPool[j])
            ++j;
        mixMat = feaMix->mixMats[j];
        for (k = 0, curOff = 0; k < feaMix->elemNum; ++k) {
            feaElem = feaMix->feaList[k];
            l = 1;
            /* 1. extended input features */
            if (feaElem->inputKind == INPFEAIK || feaElem->inputKind == AUGFEAIK) {
                if (feaElem->inputKind == INPFEAIK) {
                    while (layerElem->drvCtx[i] != feaElem->ctxPool[l])
                        ++l;
                }
                feaMat = feaElem->feaMats[l];
                for (t = 0, srcOff = 0, dstOff = curOff; t < batLen; ++t, srcOff += feaElem->extDim, dstOff += feaMix->mixDim) {
                    CopyNSegment(feaMat, srcOff, feaElem->extDim, mixMat, dstOff);
                }
            }
            else if (feaElem->inputKind == ANNFEAIK) {	/* 2. ANN features */
                m = IntVecSize(feaElem->ctxMap);
                for (c = 1; c <= m; ++c) {
                    curCtx = feaElem->ctxMap[c] + layerElem->drvCtx[i];
                    while (curCtx != feaElem->ctxPool[l])
                        ++l;
                    feaMat = feaElem->feaMats[l];
                    for (t = 0; t < batLen; ++t) {
                        srcOff = t * feaElem->srcDim + feaElem->dimOff;
                        dstOff = t * feaMix->mixDim + curOff + (c - 1) * feaElem->feaDim;
                        CopyNSegment(feaMat, srcOff, feaElem->feaDim, mixMat, dstOff);
                    }
                }
            }
            curOff += feaElem->extDim;
        }
    }
}


/* fill a batch with error signal */
static inline void FillBatchFromErrMixOLD(FeaMix *errMix, int batLen, NMatrix *mixMat) {
    int i, j, srcOff, dstOff;
    FELink errElem;

    /* if it is the shared */
    if (errMix->feaList[0]->feaMats[1] == mixMat) {
        return;
    }

    /* otherwise, fill the batch with a mixture of the FeaElem */
    dstOff = 0;
    /* reset mixMat to 0 */
    /*SetNMatrix(0.0, mixMat, batLen);*/
    ClearNMatrix(mixMat, batLen);
    /* accumulate the error signals from each source */
    for (i = 0; i < batLen; ++i) {
        for (j = 0; j < errMix->elemNum; ++j) {
            errElem = errMix->feaList[j];
            srcOff = i * errElem->srcDim + errElem->dimOff;
            AddNSegment(errElem->feaMats[1], srcOff, errElem->extDim, mixMat, dstOff);
            dstOff += errElem->extDim;
        }
    }
}

/* cz277 - many */
static inline void FillBatchFromErrMix(LELink layerElem, int batLen) {
    int c, i, j, k, l, m, n, t, srcOff, dstOff;
    FELink errElem;
    FeaMix *errMix;

    if (layerElem->errMix == NULL)
        return;
    errMix = layerElem->errMix;
    if (errMix->elemNum == 1) {
        errElem = errMix->feaList[0];
        if (IntVecSize(errElem->ctxMap) == 1)
            if (errElem->srcDim == errElem->feaDim)
                return;
    }
    
    n = IntVecSize(layerElem->drvCtx);
    for (i = 1; i <= n; ++i) {
        ClearNMatrix(errMix->mixMats[i], batLen);
    }
    for (i = 0; i < errMix->elemNum; ++i) {
        errElem = errMix->feaList[i];
        m = IntVecSize(errElem->ctxPool);
        n = IntVecSize(errElem->ctxMap);
        for (j = 1; j <= m; ++j) {
            for (k = 1; k <= n; ++k) {
                srcOff = errElem->dimOff + (k - 1) * errElem->feaDim;
                c = errElem->ctxPool[j] + errElem->ctxMap[k];
                l = 1;
                while (errMix->ctxPool[l] != c)
                    ++l;
                for (t = 0, dstOff = 0; t < batLen; ++t, srcOff += errElem->srcDim, dstOff += errElem->feaDim) {
                    AddNSegment(errElem->feaMats[j], srcOff, errElem->feaDim, errMix->mixMats[l], dstOff);
                }
            }
        }
    }
    /* scale the mixMats */
    /*n = IntVecSize(layerElem->drvCtx);
    for (i = 1; i <= n; ++i) {
        if (layerElem->trainInfo->drvCnt[i] > 1)
            ScaleNMatrix(1.0 / (float) layerElem->trainInfo->drvCnt[i], batLen, errMix->mixDim, errMix->mixMats[i]);
    }*/

}


/* temp function */
void ShowAddress(ANNSet *annSet) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    curAI = annSet->defsHead;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        printf("ANNInfo = %p. ANNDef = %p: \n", curAI, annDef);
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            /*printf("layerElem = %p, feaMix[0]->feaMat = %p, xFeaMat = %p, yFeaMat = %p, trainInfo = %p, dxFeaMat = %p, dyFeaMat = %p, labMat = %p\n", layerElem, layerElem->feaMix->feaList[0]->feaMat, layerElem->xFeaMat, layerElem->yFeaMat, layerElem->trainInfo, layerElem->trainInfo->dxFeaMat, layerElem->trainInfo->dyFeaMat, layerElem->trainInfo->labMat);*/
        }
        printf("\n");
        curAI = curAI->next;
    }
}

/* update the map sum matrix for outputs */
void UpdateOutMatMapSum(ANNSet *annSet, int batLen, int streamIdx) {

    /* cz277 - many */
    HNBlasTNgemm(annSet->mapStruct->mappedTargetNum, batLen, annSet->outLayers[streamIdx]->nodeNum, 1.0, annSet->mapStruct->maskMatMapSum[streamIdx], annSet->outLayers[streamIdx]->yFeaMats[1], 0.0, annSet->mapStruct->outMatMapSum[streamIdx]);
}

/* update the map sum matrix for labels */
void UpdateLabMatMapSum(ANNSet *annSet, int batLen, int streamIdx) {

    HNBlasTNgemm(annSet->mapStruct->mappedTargetNum, batLen, annSet->outLayers[streamIdx]->nodeNum, 1.0, annSet->mapStruct->maskMatMapSum[streamIdx], annSet->outLayers[streamIdx]->trainInfo->labMat, 0.0, annSet->mapStruct->labMatMapSum[streamIdx]);
}

/* cz277 - pact */
/* y = 1 / sqrt(var) * x + (- mean / sqrt(var)) */
static void InitAffineScaleByVar(int vecLen, NVector *varVec) {
    int i;
  
    if (!(vecLen <= varVec->vecLen)) {
        HError(9999, "InitAffineScaleByVar: Wrong vector length");
    }
#ifdef CUDA
    SyncNVectorDev2Host(varVec);
#endif
    /* convert variance */
    for (i = 0; i < vecLen; ++i) {
        if (varVec->vecElems[i] <= 0.0) {
            HError(9999, "InitAffineScaleByVar: variance should be > 0.0");
        }
        varVec->vecElems[i] = 1.0 / sqrt(varVec->vecElems[i]);
    }
#ifdef CUDA
    SyncNVectorHost2Dev(varVec);
#endif
}

/* cz277 - pact */
/* y = 1 / sqrt(var) * x + (- mean / sqrt(var)) */
static void InitAffineShiftByMean(int vecLen, NVector *scaleVec, NVector *meanVec) {
    int i;
    
    if (!(vecLen <= meanVec->vecLen && vecLen <= scaleVec->vecLen)) {
        HError(9999, "InitAffineShiftByMean: Wrong vector length");
    }
#ifdef CUDA
    SyncNVectorDev2Host(meanVec);
    SyncNVectorDev2Host(scaleVec);
#endif
    /* convert mean */
    for (i = 0; i < vecLen; ++i) {
        meanVec->vecElems[i] = (-1.0) * meanVec->vecElems[i] * scaleVec->vecElems[i];
    }
#ifdef CUDA
    SyncNVectorHost2Dev(meanVec);
    SyncNVectorHost2Dev(scaleVec);
#endif
}

/* cz277 - pact */
void DoStaticUpdateOperation(int status, int drvIdx, LELink layerElem, int batLen) {
    double nSamples;
    size_t *cnt1, *cnt2;

    if (layerElem->trainInfo == NULL)
        return;
    if ((layerElem->trainInfo->updtFlag & INITUK) == 0)
        return;

    switch (layerElem->actFunInfo.actFunKind) {
    case AFFINEAF:
        if (layerElem->drvCtx[drvIdx] == 0) {
            cnt1 = (size_t *) layerElem->actFunInfo.actParmVec[1]->hook;
            cnt2 = (size_t *) layerElem->actFunInfo.actParmVec[2]->hook;
            switch (status) {
	    case 0:
	        *cnt1 += batLen;
                *cnt2 += batLen;	
		break;
            case 1:
		nSamples = *cnt2;
                if (nSamples == 0)
                    HError(-9999, "DoStaticUpdateOperation: nSamples = 0, inf will generate");
                AccMeanNVector(layerElem->yFeaMats[drvIdx], batLen, layerElem->nodeNum, (NFloat) nSamples, layerElem->actFunInfo.actParmVec[2]);
                break;
            case 2:
                nSamples = *cnt1;
                if (nSamples == 0)
                    HError(-9999, "DoStaticUpdateOperation: nSamples = 0, inf will generate");
                AccVarianceNVector(layerElem->yFeaMats[drvIdx], batLen, layerElem->nodeNum, (NFloat) nSamples, layerElem->actFunInfo.actParmVec[2], layerElem->actFunInfo.actParmVec[1]);
                break;
            case 3:
                nSamples = *cnt1;
                if (nSamples > 0) {
                    InitAffineScaleByVar(layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1]);
                    *cnt1 = 0;
                }
                break;
            case 4:
                nSamples = *cnt2;
                if (nSamples > 0) {
                    InitAffineShiftByMean(layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2]);
                    *cnt2 = 0;
                }
                break;
            default:
                break;
            }
        }
        break;
    default:
        break;
    }
}

/* cz277 - pact */
void ForwardPropBlank(ANNSet *annSet, int batLen) {
    int i, j, n;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    /* init the ANNInfo pointer */
    curAI = annSet->defsHead;
    /* proceed in the forward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = 0; i < annDef->layerNum; ++i) {
            /* get current LayerElem */
            layerElem = annDef->layerList[i];
            n = IntVecSize(layerElem->drvCtx);
            for (j = 1; j <= n; ++j) {
                DoStaticUpdateOperation(layerElem->status, j, layerElem, batLen);
            }
        }
        curAI = curAI->next;
    }
}

/* the batch with input features are assumed to be filled */
void ForwardPropBatch(ANNSet *annSet, int batLen, int *CMDVecPL) {
    int i, j, n;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    /* init the ANNInfo pointer */
    curAI = annSet->defsHead;
    /* proceed in the forward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;


        //cw564 - stimu -- begin
        if (STP()->phonepos == NULL) {
            ParsePhonePos();
        }
        
        if (!STP()->acti_surface) {
            int dim = annDef->layerList[0]->nodeNum - STP()->mixnodes;
            STP()->grid_row = STP()->grid_col = (int)sqrt(dim);
            STP()->acti_surface = CreateNMatrix(STP()->heap, dim, STP()->num_phone);
            STP()->mix_surface = CreateNMatrix(STP()->heap, STP()->num_phone, dim);

            STP()->pos_dist = CreateNMatrix(STP()->heap, STP()->num_phone, dim);

            if (STP()->grid_row * STP()->grid_col != dim) {
                HError(9999, "HNSTIMU: Wrong hidden dim %d.", dim);
            }
            int p,q,r;
            for (q = 0; q < STP()->num_phone; ++ q) {
                float Z = 0;
                float maxVal = 0;
                for (r = 0; r < dim; ++ r) {
                    int row_no = r / STP()->grid_row;
                    int col_no = r % STP()->grid_row;
                    float x = (0.5 + col_no) / STP()->grid_col;
                    float y = (0.5 + row_no) / STP()->grid_row;
                    float dissq =
                            (x - STP()->phonepos->matElems[q * 2]) * (x - STP()->phonepos->matElems[q * 2])
                            + (y - STP()->phonepos->matElems[q * 2 + 1]) * (y - STP()->phonepos->matElems[q * 2 + 1]);
                    STP()->acti_surface->matElems[r * STP()->num_phone + q] = exp(-0.5 * dissq / STP()->grid_var);
                    STP()->mix_surface->matElems[q * dim + r] = STP()->acti_surface->matElems[r * STP()->num_phone + q];
                    STP()->pos_dist->matElems[q * dim + r] = sqrt(dissq);
                    Z += STP()->acti_surface->matElems[r * STP()->num_phone + q];
                    if (maxVal < STP()->mix_surface->matElems[q * dim + r]) {
                        maxVal = STP()->mix_surface->matElems[q * dim + r];
                    }
                }
                for (r = 0; r < dim; ++ r) {
                     STP()->acti_surface->matElems[r * STP()->num_phone + q] /= Z;
                     STP()->mix_surface->matElems[q * dim + r] /= maxVal;
                }
            }
#ifdef CUDA
            SyncNMatrixHost2Dev(STP()->acti_surface);
            SyncNMatrixHost2Dev(STP()->mix_surface);
#endif
        }
        if (!STP()->lhuc_surface) {
            int dim = annDef->layerList[0]->nodeNum - STP()->mixnodes;
            STP()->lhuc_surface = CreateNMatrix(STP()->heap, dim, dim);
            int p,q,r;
            for (q = 0; q < dim; ++ q) {
                float Z = 0;
                int q_row_no = q / STP()->grid_row;
                int q_col_no = q % STP()->grid_row;
                float q_x = (0.5 + q_col_no) / STP()->grid_col;
                float q_y = (0.5 + q_row_no) / STP()->grid_row;
                for (r = 0; r < dim; ++ r) {
                    int r_row_no = r / STP()->grid_row;
                    int r_col_no = r % STP()->grid_row;
                    float r_x = (0.5 + r_col_no) / STP()->grid_col;
                    float r_y = (0.5 + r_row_no) / STP()->grid_row;
                    float dissq = (r_x - q_x) * (r_x - q_x) + (r_y - q_y) * (r_y - q_y);
                    STP()->lhuc_surface->matElems[q * dim + r] = exp(-0.5 * dissq / STP()->lhuc_range_var);
                    Z += STP()->lhuc_surface->matElems[q * dim + r];
                }
                for (r = 0; r < dim; ++ r) {
                    STP()->lhuc_surface->matElems[q * dim + r] /= Z;
                }
                /*
                if (q == 512 + 16) {
                    for (r = 0; r < dim; ++ r) {
                        printf("%f ", STP()->lhuc_surface->matElems[q * dim + r]);
                    }
                    printf("\n");
                    exit(0);
                }
                exit(0);
                */
            }
#ifdef CUDA
            SyncNMatrixHost2Dev(STP()->lhuc_surface);
#endif
        }
        if (STP()->weight_norms == NULL) {
            STP()->weight_norms = CreateNVector(STP()->heap, annDef->layerList[0]->nodeNum - STP()->mixnodes);
        }

        if (STP()->klvec == NULL) {
            STP()->klvec = CreateNVector(STP()->heap, 1);
        }

        if (STP()->lhuc_regVal == NULL) {
            STP()->lhuc_regVal = CreateNVector(STP()->heap, 1);
        }

        if (STP()->sum_xFeaMat == NULL) {
            STP()->sum_xFeaMat = CreateNVector(STP()->heap, GetNBatchSamples());
        }
        //cw564 - stimu -- end




 
        /* proceed layer by layer */
        for (i = 0; i < annDef->layerNum; ++i) {
            /* get current LayerElem */
            layerElem = annDef->layerList[i];
            n = IntVecSize(layerElem->drvCtx);
            FillBatchFromFeaMix(layerElem, batLen);
            for (j = 1; j <= n; ++j) {
/*erray9*/
/*if (j == 1) {
printf("\n\nINPUT, layer = %d\n\n", i);
ShowNMatrix(layerElem->xFeaMats[j], 1);
printf("\n\n");
}
if (i == 2) {
exit(1);
}*/
                /* at least the batch (feaMat) for each FeaElem is already */
                /*FillBatchFromFeaMix(layerElem, batLen, j);*/
                /* do the operation of current layer */
                switch (layerElem->operKind) {
                case MAXOK:

                    break;
                case SUMOK: 
                    /* y = b, B^T should be row major matrix, duplicate the bias vectors */ 
                    DupNVector(layerElem->biasVec, layerElem->yFeaMats[j], batLen);
                    /* y += w * b, X^T is row major, W^T is column major, Y^T = X^T * W^T + B^T */
                    if (i == 0) {
                        HNBlasTNgemm(layerElem->nodeNum, batLen, layerElem->inputDim, 1.0, layerElem->wghtMat, layerElem->xFeaMats[j], 1.0, layerElem->yFeaMats[j]);
                    }
                    else {
                        /*
                        if (i == 1) {
                            printf("layernode=%d,layerinput=%d\n", layerElem->nodeNum,layerElem->inputDim);
                            printf("cbynode=%d\n,cbybatlen=%d,batlen=%d\n", STP()->comb_yFeaMat[i - 1]->colNum, STP()->comb_yFeaMat[i - 1]->rowNum, batLen);
                            printf("weightcol=%d,weightrow=%d\n", layerElem->wghtMat->colNum, layerElem->wghtMat->rowNum);
                        //    exit(0);
                        }
                        */
                        //HNBlasTNgemm(layerElem->nodeNum, batLen, layerElem->inputDim, 1.0, layerElem->wghtMat, layerElem->xFeaMats[j], 1.0, layerElem->yFeaMats[j]);
                        HNBlasTNgemm(layerElem->nodeNum, batLen, layerElem->inputDim, 1.0, layerElem->wghtMat, STP()->comb_yFeaMat[i - 1], 1.0, layerElem->yFeaMats[j]);
                    }
                    break;
                case PRODOK:

                    break;
                default:
                    HError(9999, "ForwardPropBatch: Unknown layer operation kind");
                }
                int k;
                /* cz277 - pact */
                DoStaticUpdateOperation(layerElem->status, j, layerElem, batLen);
                /* apply activation transformation */
                switch (layerElem->actFunInfo.actFunKind) {
                case AFFINEAF:
                    ApplyAffineAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2], layerElem->yFeaMats[j]);	/* cz277 - pact */
                    break;
                case HERMITEAF:
                    ApplyHermiteAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->yFeaMats[j]);
                    break;
                case LINEARAF:
                    break;
                case RELUAF:
                    ApplyReLUAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, 0.0, layerElem->yFeaMats[j]);
                    break;
                case LHUCRELUAF:
                    HError(9999, "LHUCRELU Not Implemented"); 
                    break;
                case PRELUAF:
                    ApplyPReLUAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->yFeaMats[j]);
                    break;
                case PARMRELUAF:
                    if (layerElem->trainInfo != NULL) {
                        CopyNSegment(layerElem->yFeaMats[j], 0, batLen * layerElem->nodeNum, layerElem->trainInfo->gradInfo->actCacheMats[j], 0);
                    }
/*erray9*/
/*printf("\n\nbefore:\n");
ShowNMatrix(layerElem->yFeaMats[j], 1);*/
                    ApplyParmReLUAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2], layerElem->yFeaMats[j]);
/*erray9*/
/*printf("\n\nafter:\n");
ShowNMatrix(layerElem->yFeaMats[j], 1);
exit(1);*/
                    break; 
                case SIGMOIDAF:
                    //mix
                    //SyncNMatrixDev2Host(layerElem->yFeaMats[j]);
                    //for (k = 1023; k < 1071; ++k) printf("%f ", layerElem->yFeaMats[j]->matElems[k]);
                    //printf("\n");
                    ApplySigmoidActStimuMix(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j], STP()->mixnodes);
                    /*if (i == 1) {
                    SyncNMatrixDev2Host(layerElem->yFeaMats[j]);
                    for (k = 1023; k < 1071; ++k) printf("%f ", layerElem->yFeaMats[j]->matElems[k]);
                    printf("\n");
                    exit(0);
                    }*/
                    //ApplySigmoidAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j]);
                    break;
                case LHUCSIGMOIDAF:
                    ApplyLHUCSigmoidAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->yFeaMats[j]);
                    break;
                case PSIGMOIDAF:
                    ApplyPSigmoidAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->yFeaMats[j]);
                    break; 
                case PARMSIGMOIDAF:
                    if (layerElem->trainInfo != NULL) {
                        CopyNSegment(layerElem->yFeaMats[j], 0, batLen * layerElem->nodeNum, layerElem->trainInfo->gradInfo->actCacheMats[j], 0);
                    }
/*chaopig*/
/*printf("\n\nBEFORE:\n");
ShowNMatrix(layerElem->yFeaMats[j], 1);*/
                    ApplyParmSigmoidAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2], layerElem->actFunInfo.actParmVec[3], layerElem->yFeaMats[j]);
/*chaopig*/
/*printf("\n\nAFTER:\n");
ShowNMatrix(layerElem->yFeaMats[j], 1);
exit(1);*/
                    break;
                case SOFTRELUAF:
                    ApplySoftReLAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j]);
                    break;
                case LHUCSOFTRELUAF:
                    HError(9999, "LHUCSOFTRELU Not Implemented");
                    break;
                case PSOFTRELUAF:
                    HError(9999, "PSOFTRELU Not Implemented");
                    break;
                case PARMSOFTRELUAF:
                    HError(9999, "PARMSOFTRELU Not Implemented");
                    break;
                case SOFTMAXAF:
                    ApplySoftmaxAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j]);
                    break;
                case TANHAF:
                    ApplyTanHAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j]);
                    break;
                default:
                    HError(9999, "ForwardPropBatch: Unknown activation function kind");
                }
                //cw564 - stimu
                
                if (i < annDef->layerNum - 1) {
                    /*if (i == 1) {
                        printf("dddd%d\n", layerElem->nodeNum - STP()->mixnodes);
                    }
                    */
                    int comb_dim = layerElem->nodeNum - STP()->mixnodes;
                    if (STP()->comb_yFeaMat[i] == NULL) {
                        STP()->comb_yFeaMat[i] = CreateNMatrix(STP()->heap, GetNBatchSamples(), comb_dim);
                        STP()->comb_softmaxSum[i] = CreateNVector(STP()->heap, GetNBatchSamples());
                    }
                    FwdCombDNNandMix(STP()->comb_yFeaMat[i], STP()->mix_surface, layerElem->yFeaMats[j], STP()->comb_softmaxSum[i],
                            batLen, layerElem->nodeNum, comb_dim, STP()->mixnodes, STP()->mixscaler);
                    /*if (i == 1) {
                        SyncNMatrixDev2Host(STP()->comb_yFeaMat[i]);
                        int k;
                        for (k = 0; k < 1024; ++ k) {
                            printf("%f ", STP()->comb_yFeaMat[i]->matElems[k]);
                        }
                        printf("\n");
                        exit(0);
                    }*/
                    CalcWeightNorms(STP()->weight_norms, annDef->layerList[i + 1]->wghtMat, annDef->layerList[i + 1]->inputDim, annDef->layerList[i + 1]->nodeNum);
                    CalcSumActi(STP()->sum_xFeaMat, STP()->comb_yFeaMat[i], STP()->weight_norms, batLen, comb_dim); //mix
                    CalcStimuKL(STP()->klvec, STP()->comb_yFeaMat[i], 
                            STP()->weight_norms, STP()->sum_xFeaMat, STP()->acti_surface, STP()->phoneidx_vec, 
                            batLen, comb_dim, STP()->num_phone); //mix
                    SyncNVectorDev2Host(STP()->klvec);
                    STP()->accumKL += STP()->klvec->vecElems[0];
                }

                //HError(9999, "DBGss%s %d", STP()->visual_fn, strcmp(STP()->visual_fn, ""));
              
                if (strcmp(STP()->visual_fn, "") && i < annDef->layerNum - 1) {
                    //HError(9999, "DBGcacan");
                    char tmp[256];
                    strcpy(tmp, STP()->visual_fn);
                    char tmp2[256];
                    tmp2[0] = '0' + i;
                    tmp2[1] = '\0';
                    strcat(tmp, tmp2);
                    FILE * tmp_f = fopen(tmp, "w");
                    CalcWeightNorms(STP()->weight_norms, annDef->layerList[i + 1]->wghtMat, annDef->layerList[i + 1]->inputDim, annDef->layerList[i + 1]->nodeNum);
#ifdef CUDA
                    SyncNVectorDev2Host(STP()->weight_norms);
                    SyncNMatrixDev2Host(STP()->comb_yFeaMat[i]);
#endif
                    int kk, jj;
                    for (kk = 0; kk < batLen; ++ kk) {
                        fprintf(tmp_f, "%s", STP()->phone_names[(int)(STP()->phoneidx_vec->vecElems[kk])]);
                        for (jj = 0; jj < layerElem->nodeNum - STP()->mixnodes; ++ jj) {
                            fprintf(tmp_f, " %f", STP()->comb_yFeaMat[i]->matElems[kk * (layerElem->nodeNum - STP()->mixnodes) + jj] * STP()->weight_norms->vecElems[jj]);
                        }
                        fprintf(tmp_f, "\n");
                    }
                    fclose(tmp_f);
                }
                //cw564 - end
            }
        }

        /* get the next ANNDef */
        curAI = curAI->next;
    }
    //cw564 - stimu
    if (strcmp(STP()->visual_fn, "")) {
        exit(0);
    }
    //cw564 - end

 
    SetBatchIndex(GetBatchIndex() + 1);

}

/* function to compute the error signal for frame level criteria (for sequence level, do nothing) */
void CalcOutLayerBackwardSignal(LELink layerElem, int batLen, ObjFunKind objfunKind, int ctxIdx) {

    if (layerElem->roleKind != OUTRK) {
        HError(9999, "CalcOutLayerBackwardSignal: Function only valid for output layers");
    }
    if (ctxIdx != 1 || layerElem->drvCtx[ctxIdx] != 0) {
        HError(9999, "CalcOutLayerBackwardSignal: Out layer only support single current frame at the moment");
    }
    switch (objfunKind) {
    case MMSEOF:
        /* proceed for MMSE objective function */
        switch (layerElem->actFunInfo.actFunKind) {
        case LINEARAF:
            break; 
        case SOFTMAXAF:
            break;
        default:
            HError(9999, "CalcOutLayerBackwardSignal: Unknown output activation function");
        }
        break;
    case XENTOF:
        /* proceed for XENT objective function */
        switch (layerElem->actFunInfo.actFunKind) {
        case LINEARAF:
            break;
        case SOFTMAXAF:
            SubNMatrix(layerElem->yFeaMats[ctxIdx], layerElem->trainInfo->labMat, batLen, layerElem->nodeNum, layerElem->yFeaMats[ctxIdx]);
            break;
        default:
            HError(9999, "CalcOutLayerBackwardSignal: Unknown output activation function");
        }
        break;
    case MLOF:
    case MMIOF:
    case MPEOF:
    case MWEOF:
    case SMBROF:
        break;
    default:
        HError(9999, "CalcOutLayerBackwardSignal: Unknown objective function kind");
    }
}

/* cz277 - gradprobe */
#ifdef GRADPROBE
void AccGradProbeWeight(LayerElem *layerElem) {
    int i, j, k, size;
    NFloat *wghtMat;

#ifdef CUDA
    SyncNMatrixDev2Host(layerElem->trainInfo->gradInfo->wghtMat);
#endif
    wghtMat = layerElem->trainInfo->gradInfo->wghtMat->matElems;
    /* weights */
    size = layerElem->nodeNum * layerElem->inputDim;
    j = DVectorSize(layerElem->wghtGradInfoVec);
    for (i = 0; i < size; ++i) {
        if (wghtMat[i] > layerElem->maxWghtGrad) 
            layerElem->maxWghtGrad = wghtMat[i];
        if (wghtMat[i] < layerElem->minWghtGrad)
            layerElem->minWghtGrad = wghtMat[i];
        layerElem->meanWghtGrad += wghtMat[i];
        k = wghtMat[i] / PROBERESOLUTE + j / 2;
        layerElem->wghtGradInfoVec[k + 1] += 1;
    }
}
#endif

/* cz277 - gradprobe */
#ifdef GRADPROBE
void AccGradProbeBias(LayerElem *layerElem) {
    int i, j, k, size;
    NFloat *biasVec;

#ifdef CUDA
    SyncNVectorDev2Host(layerElem->trainInfo->gradInfo->biasVec);
#endif
    biasVec = layerElem->trainInfo->gradInfo->biasVec->vecElems;
    /* biases */
    size = layerElem->nodeNum;
    j = DVectorSize(layerElem->biasGradInfoVec);
    for (i = 0; i < size; ++i) {
        if (biasVec[i] > layerElem->maxBiasGrad)
            layerElem->maxBiasGrad = biasVec[i];
        if (biasVec[i] < layerElem->minBiasGrad)
            layerElem->minBiasGrad = biasVec[i];
        layerElem->meanBiasGrad += biasVec[i];
        k = biasVec[i] / PROBERESOLUTE + j / 2;
        layerElem->biasGradInfoVec[k + 1] += 1;
    }
}
#endif

/* delta_j = h'(a_j) * sum_k [w_k,j * delta_k] */
/*   dtl_j = sum_k [w_k,j * dtl_k * h'(a_k)] */
/*   dtl_j = delta_j / h'(a_j) */
/* backward propagation algorithm */
void BackwardPropBatch(ANNSet *annSet, int batLen, Boolean accGrad) {
    int i, j, n;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    NMatrix *dyFeaMat;
    Boolean accFlag;

    /* init the ANNInfo pointer */
    curAI = annSet->defsTail;
    /* proceed in the backward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            /* get current LayerElem */
            layerElem = annDef->layerList[i];
            n = IntVecSize(layerElem->drvCtx);
            FillBatchFromErrMix(layerElem, batLen);	/* cz277 - many */
            for (j = 1; j <= n; ++j) {
                if (j == 1) 
                    accFlag = accGrad;
                else
                    accFlag = TRUE;
                /* proceed different types of layers */
                if (layerElem->roleKind == OUTRK) {
                    /* set dyFeaMat */
                    CalcOutLayerBackwardSignal(layerElem, batLen, annDef->objfunKind, j);
                    dyFeaMat = layerElem->yFeaMats[j];	/* delta_k */
                }
                else {
                    /* at least the batch (feaMat) for each FeaElem is already */
                    /*FillBatchFromErrMix(layerElem, batLen, j);*/	/* cz277 - many */

                    //cw564 - stimu -- begin
                    
                    //mix
                    int comb_dim = layerElem->nodeNum - STP()->mixnodes;
                    //dyFeaMat = layerElem->trainInfo->dyFeaMats[j];	/* sum_k w_{k,j} * delta_k */
                    dyFeaMat = STP()->comb_dxFeaMat; //TODO name not nice
                    SyncNMatrixDev2Host(dyFeaMat);
                    int k;
                    /*for (k = 0; k < 10; ++ k) {
                        printf("%f ", dyFeaMat->matElems[k]);
                    }
                    printf("\n");exit(0);
                    */
                    if (STP()->stimu_dxFeaMat == NULL) {
                        STP()->stimu_dxFeaMat = CreateNMatrix(STP()->heap, GetNBatchSamples(), comb_dim); //mix
                    }
                    
                    if (STP()->lhuc_penalty_type == 0) {
                        CalcWeightNorms(STP()->weight_norms, annDef->layerList[i + 1]->wghtMat,
                                annDef->layerList[i + 1]->inputDim, annDef->layerList[i + 1]->nodeNum);
                        //printf("nextinpdim=%d,combdim=%d,layerdim=%d\n", annDef->layerList[i + 1]->inputDim, comb_dim, layerElem->nodeNum); exit(0);
                        StimuGrad(STP()->comb_yFeaMat[i], 
                                batLen, comb_dim, STP()->acti_surface, STP()->num_phone, STP()->phoneidx_vec, STP()->stimu_dxFeaMat, STP()->weight_norms);//mix
                    
                        AddScaledNMatrix(STP()->stimu_dxFeaMat, batLen, comb_dim, STP()->stimu_penalty, dyFeaMat);
                    }
                    //split dyFeaMat to mix and dnn nodes
                    SplitGradDNNandMix(layerElem->trainInfo->dyFeaMats[j], dyFeaMat, STP()->mix_surface, batLen, layerElem->nodeNum, comb_dim, STP()->mixscaler);
                    dyFeaMat = layerElem->trainInfo->dyFeaMats[j];
                    //exit(0);
                    //cw564 - stimu -- end
 


                    /* cz277 - pact, compute the graident for the trainable hidden activation function gradients here */
                    switch (layerElem->actFunInfo.actFunKind) {
                    case AFFINEAF:
                        ApplyTrAffineAct(dyFeaMat, layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2], accFlag, layerElem->trainInfo->gradInfo->actParmVec[1], layerElem->trainInfo->gradInfo->actParmVec[2]);
                        break;
                    case HERMITEAF:
			HError(9999, "HERMITE Not Implemented yet");
                        break;
                    case LHUCRELUAF:
                        HError(9999, "LHUCRELU Not Implemented yet");
                        break;
                    case PRELUAF:
                        ApplyTrPReLUAct(dyFeaMat, layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], accFlag, layerElem->trainInfo->gradInfo->actParmVec[1]);
                        break;
                    case PARMRELUAF:
/*erray9*/
/*printf("\n\ndyFeaMat:\n");
ShowNMatrix(dyFeaMat, 1);
printf("\n\nactCacheMats\n");
ShowNMatrix(layerElem->trainInfo->gradInfo->actCacheMats[j], 1);*/
                        ApplyTrParmReLUAct(dyFeaMat, layerElem->trainInfo->gradInfo->actCacheMats[j], batLen, layerElem->nodeNum, accFlag, layerElem->trainInfo->gradInfo->actParmVec[1], layerElem->trainInfo->gradInfo->actParmVec[2]);
/*erray9*/
/*printf("\n\nactParmVec[1]\n");
ShowNVector(layerElem->trainInfo->gradInfo->actParmVec[1]);
printf("\n\nactParmVec[2]\n");
ShowNVector(layerElem->trainInfo->gradInfo->actParmVec[2]);
exit(1);*/
                        break;
                    case LHUCSIGMOIDAF:
                        ApplyTrLHUCSigmoidAct(dyFeaMat, layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], accFlag, layerElem->trainInfo->gradInfo->actParmVec[1]);
                        break;
                    case PSIGMOIDAF:
                        ApplyTrPSigmoidAct(dyFeaMat, layerElem->yFeaMats[j], layerElem->actFunInfo.actParmVec[1], batLen, layerElem->nodeNum, accFlag, layerElem->trainInfo->gradInfo->actParmVec[1]);
                        break;
                    case PARMSIGMOIDAF:
                        ApplyTrParmSigmoidAct(dyFeaMat, layerElem->trainInfo->gradInfo->actCacheMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2], layerElem->actFunInfo.actParmVec[3], accFlag, layerElem->trainInfo->gradInfo->actParmVec[1], layerElem->trainInfo->gradInfo->actParmVec[2], layerElem->trainInfo->gradInfo->actParmVec[3]);
                        break;
                    case LHUCSOFTRELUAF:
                        HError(9999, "LHUCSOFTRELU Not Implemented yet");
                        break;
                    case PSOFTRELUAF:
                        HError(9999, "PSOFTRELU Not Implemented yet");
                        break;
                    case PARMSOFTRELUAF:
                        HError(9999, "PARMSOFTRELU Not Implemented yet");
                        break;
                    default:
                        break;
                    }
                    
                    //cw564 - stimu
                    if (STP()->lhuc_penalty_type != 0) {
                        printf("MIX ADAPT NOT SUPPORT\n");
                        //ShowNVector(layerElem->trainInfo->gradInfo->actParmVec[1]);
                        //ApplyActStimuPenalty(STP()->stimu_dxFeaMat, layerElem->yFeaMats[j], layerElem->actFunInfo.actParmVec[1], layerElem->trainInfo->gradInfo->actParmVec[1], batLen, layerElem->nodeNum, STP()->stimu_penalty, accFlag);
                        //ShowNVector(layerElem->trainInfo->gradInfo->actParmVec[1]);
                        //HError(9999, "");
                        //if ()
                        if (STP()->lhuc_penalty_type == 1) {
                            if (STP()->d_lhuc == NULL) {
                                STP()->d_lhuc = CreateNMatrix(STP()->heap, GetNBatchSamples(), layerElem->nodeNum);
                            }
                            
                            STP()->lhuc_regVal->vecElems[0] = 0;
                            SyncNVectorHost2Dev(STP()->lhuc_regVal);
                            //ShowNVector(layerElem->trainInfo->gradInfo->actParmVec[1]);
                            ApplyActLHUCPenaltyLocalSoft(layerElem->actFunInfo.actParmVec[1], layerElem->trainInfo->gradInfo->actParmVec[1], 
                                STP()->lhuc_surface, layerElem->nodeNum, batLen, STP()->lhuc_penalty, STP()->lhuc_regVal);
                            //ShowNVector(layerElem->trainInfo->gradInfo->actParmVec[1]);
                            SyncNVectorDev2Host(STP()->lhuc_regVal);
                            STP()->accumLHUCRegVal += STP()->lhuc_regVal->vecElems[0];
                        }
                    }
                    
                    //cw564 - end
                    /* apply activation transformation */
                    switch (layerElem->actFunInfo.actFunKind) {
                    case AFFINEAF:
                        ApplyDAffineAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2], layerElem->yFeaMats[j]);	/* h(a_j) -> h'(a_j) */
                        break;
                    case HERMITEAF:
                        HError(9999, "HERMITE Not Implemented yet");
                        break;
                    case LINEARAF:
                        ApplyDLinearAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j]);
                        break;
                    case RELUAF:
                        ApplyDReLUAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, 0.0, layerElem->yFeaMats[j]);
                        break;
                    case LHUCRELUAF:
                        HError(9999, "LHUCRELU Not Implemented yet");
                        break;
                    case PRELUAF:
                        ApplyDPReLUAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->yFeaMats[j]);
                        break;
                    case PARMRELUAF:
/*erray9*/
/*printf("\n\nafter D:\n");
ShowNMatrix(layerElem->yFeaMats[j], 1);*/
                        ApplyDParmReLUAct(layerElem->trainInfo->gradInfo->actCacheMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2], layerElem->yFeaMats[j]);
/*erray9*/
/*printf("\n\nafter D:\n");
ShowNMatrix(layerElem->yFeaMats[j], 1);
exit(1);*/
                        break;
                    case SIGMOIDAF:
                        /*SyncNMatrixDev2Host(layerElem->yFeaMats[j]);
                        SyncNVectorDev2Host(STP()->comb_softmaxSum[i]);
                        for (k = 1023; k < 1071; ++ k) {
                            printf("%f ", layerElem->yFeaMats[j]->matElems[k]);
                        }
                        printf("\n");
                        printf("%f\n", STP()->comb_softmaxSum[i]->vecElems[0]);
                        exit(0);
                        */
                        //ApplyDSigmoidActStimuMix(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j], STP()->mixnodes);//mix
                        ApplyDSigmoidAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j]);
                        break;
                    case LHUCSIGMOIDAF:
                        ApplyDLHUCSigmoidAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->yFeaMats[j]);
                        break;
                    case PSIGMOIDAF:
                        ApplyDPSigmoidAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->yFeaMats[j]);
                        break;
                    case PARMSIGMOIDAF:
                        ApplyDParmSigmoidAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->actFunInfo.actParmVec[1], layerElem->actFunInfo.actParmVec[2], layerElem->actFunInfo.actParmVec[3], layerElem->yFeaMats[j]);
                        break;
                    case SOFTRELUAF:
                        ApplyDSoftReLAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j]);
                        break;
                    case LHUCSOFTRELUAF:
                        HError(9999, "LHUCSOFTRELU Not Implemented yet!");
                        break;
                    case PSOFTRELUAF:
                        HError(9999, "PSOFTRELU Not Implemented yet!");
                        break;
                    case PARMSOFTRELUAF:
                        HError(9999, "PARMSOFTRELU Not Implemented yet!");
                        break;
                    case SOFTMAXAF:
                        HError(9999, "SOFTMAX Not Implemented yet!");
                        break;
                    case TANHAF:
                        ApplyDTanHAct(layerElem->yFeaMats[j], batLen, layerElem->nodeNum, layerElem->yFeaMats[j]);
                        break;
                    default:
                        HError(9999, "BackwardPropBatch: Unknown hidden activation function kind");
                    }
                    /* times sigma_k (dyFeaMat, from the next layer) */
                    /* dyFeaMat: sum_k w_{k,j} * delta_k -> delta_j */
                    MulNMatrix(layerElem->yFeaMats[j], dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);	/* delta_j = h'(a_j) * (sum_k w_{k,j} * delta_k) */
                }
                /* do current layer operation */
                //cw564 - stimu -- mix
                int comb_dim = annDef->layerList[1]->inputDim;
                //printf("combdimforbp=%d\n", comb_dim);
                int raw_dim = layerElem->inputDim + STP()->mixnodes;
                if (STP()->comb_dxFeaMat == NULL) {
                    STP()->comb_dxFeaMat = CreateNMatrix(STP()->heap, GetNBatchSamples(), comb_dim);
                }


                switch (layerElem->operKind) {
                case MAXOK:

                    break;
                case SUMOK:
                    /* Y^T is row major, W^T is column major, X^T = Y^T * W^T */
                    HNBlasNNgemm(layerElem->inputDim, batLen, layerElem->nodeNum, 1.0, layerElem->wghtMat, dyFeaMat, 0.0, STP()->comb_dxFeaMat);	/* sum_k w_{k,j} * delta_k */
                    //layerElem->trainInfo->dxFeaMats[j];
                    break;
                case PRODOK:

                    break;
                default:
                    HError(9999, "BackwardPropBatch: Unknown layer operation kind");
                }
                /*  SyncNMatrixDev2Host(STP()->comb_dxFeaMat);
                    int k;
                    for (k = 0; k < 10; ++ k) {
                        printf("%f ", STP()->comb_dxFeaMat->matElems[k]);
                    }
                    printf("\n");
                */
                /* compute and accumulate the updates */
                /* {layerElem->xFeaMat[n_frames * inputDim]}^T * dyFeaMat[n_frames * nodeNum] = deltaWeights[inputDim * nodeNum] */
                if (layerElem->trainInfo->updtFlag & WEIGHTUK) {
                    if (i == 0) {
                        HNBlasNTgemm(layerElem->inputDim, layerElem->nodeNum, batLen, 1.0, layerElem->xFeaMats[j], dyFeaMat, accFlag, layerElem->trainInfo->gradInfo->wghtMat);
                    }
                    else {
                        HNBlasNTgemm(layerElem->inputDim, layerElem->nodeNum, batLen, 1.0, STP()->comb_yFeaMat[i - 1], dyFeaMat, accFlag, layerElem->trainInfo->gradInfo->wghtMat);
                    }
                    /* cz277 - gradprobe */
#ifdef GRADPROBE
                    AccGradProbeWeight(layerElem);    
#endif
                }
                /* graidents for biases */
                if (layerElem->trainInfo->updtFlag & BIASUK) {
                    SumNMatrixByCol(dyFeaMat, batLen, layerElem->nodeNum, accFlag, layerElem->trainInfo->gradInfo->biasVec);
                    /* cz277 - gradprobe */
#ifdef GRADPROBE
                    AccGradProbeBias(layerElem);
#endif
                }

                if (layerElem->trainInfo->ssgInfo != NULL) {
                    /* attention: these two operations are gonna to change dyFeaMat elements to their square */
                    SquaredNMatrix(layerElem->xFeaMats[j], batLen, layerElem->inputDim, GetTmpNMat());
                    SquaredNMatrix(dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);
                    if (layerElem->trainInfo->updtFlag & WEIGHTUK)
                        HNBlasNTgemm(layerElem->inputDim, layerElem->nodeNum, batLen, 1.0, GetTmpNMat(), dyFeaMat, 1.0, layerElem->trainInfo->ssgInfo->wghtMat);
                    if (layerElem->trainInfo->updtFlag & BIASUK) 
                        SumNMatrixByCol(dyFeaMat, batLen, layerElem->nodeNum, TRUE, layerElem->trainInfo->ssgInfo->biasVec);
                }
            }
        }

        /* get the previous ANNDef */
        curAI = curAI->prev;
    }

}

/* randomise an ANN layer */
void RandANNLayer(LELink layerElem, int seed, float scale) {
    float r;

    switch (layerElem->actFunInfo.actFunKind) {
    case AFFINEAF:
    case LINEARAF:
    case RELUAF:
    case LHUCRELUAF:
    case PRELUAF:
    case PARMRELUAF:
    case SOFTRELUAF:
    case LHUCSOFTRELUAF:
    case PSOFTRELUAF:
    case PARMSOFTRELUAF:
        r = 16.0 / ((float) (layerElem->nodeNum + layerElem->inputDim));
	/* 0.004 for a (2000, 2000) layer; r = 0.001 for a (12000, 2000) layer */	
        r *= scale;
        RandInit(seed);
        RandNSegmentUniform(-1.0 * r, r, layerElem->wghtMat->rowNum * layerElem->wghtMat->colNum, layerElem->wghtMat->matElems);
        break;
        /*r = sqrt(2.0 / ((1.0 + pow(PLRELUNEGSCALE, 2.0)) * layerElem->nodeNum));     
        RandNSegmentGaussian(0.0, r, layerElem->wghtMat->rowNum * layerElem->wghtMat->colNum, layerElem->wghtMat->matElems);*/
    default:
        r = 4 * sqrt(6.0 / (float) (layerElem->nodeNum + layerElem->inputDim));
        r *= scale;
        RandInit(seed);
        RandNSegmentUniform(-1.0 * r, r, layerElem->wghtMat->rowNum * layerElem->wghtMat->colNum, layerElem->wghtMat->matElems);
	/* r = 0.22 for a (1000, 1000) layer; r = 0.083 for a (12000, 2000) layer */
        break;
    }

    /*if (layerElem->actfunKind == RELAF || layerElem->actfunKind == SOFTRELAF) {
        RandMaskNSegment(0.25, 0.0, layerElem->wghtMat->rowNum * layerElem->wghtMat->colNum, layerElem->wghtMat->matElems);
    }*/

    ClearNVector(layerElem->biasVec);
    /* TODO: if HERMITEAF */
#ifdef CUDA
    SyncNMatrixHost2Dev(layerElem->wghtMat);
    SyncNVectorDev2Host(layerElem->biasVec);
#endif

}

/* generate a new ANN layer and randomise it */
/*LELink GenRandLayer(MemHeap *heap, int nodeNum, int inputDim, int seed) {*/
LELink GenNewLayer(MemHeap *heap, int nodeNum, int inputDim) {
     LELink layerElem;

     /*layerElem = (LELink) New(heap, sizeof(LayerElem));*/
     layerElem = GenBlankLayer(heap);
     /*layerElem->operKind = operKind;
     layerElem->actfunKind = actfunKind;*/
     layerElem->nodeNum = nodeNum;
     layerElem->inputDim = inputDim;
     layerElem->wghtMat = CreateNMatrix(heap, nodeNum, inputDim);
     layerElem->biasVec = CreateNVector(heap, nodeNum);

     /*RandANNLayer(layerElem, seed);*/

     return layerElem;     
}

void SetFeaMixBatchIdxes(ANNSet *annSet, int newIdx) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    NMatrix *dyFeaMat;

    /* init the ANNInfo pointer */
    curAI = annSet->defsTail;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            layerElem = annDef->layerList[i];
	    if (layerElem->feaMix->batIdx == 0) {
                layerElem->feaMix->batIdx = newIdx; 
            }
        }
        curAI = curAI->next;
    }
}

/* cz277 - max norm2 */
Boolean IsLinearActFun(ActFunKind actfunKind) {
    switch (actfunKind) {
        case AFFINEAF:
        case LINEARAF:
        case RELUAF:
        case LHUCRELUAF:
        case PRELUAF:
        case PARMRELUAF:
        case SOFTRELUAF:
            return TRUE;
        default:
            return FALSE;
    }
}

/* cz277 - max norm2 */
Boolean IsNonLinearActFun(ActFunKind actfunKind) {
    switch (actfunKind) {
        case HERMITEAF:
        case SIGMOIDAF:
        case LHUCSIGMOIDAF:
        case PSIGMOIDAF:
        case PARMSIGMOIDAF:
        case SOFTRELUAF:
        case LHUCSOFTRELUAF:
        case PSOFTRELUAF:
        case PARMSOFTRELUAF:
        case SOFTMAXAF:
        case TANHAF:
            return TRUE;
        default:
            return FALSE;
    }
}

/* cz277 - xform */
char *MakeRplNMatName(char *curSpkr, int rplIdx, char *rplName) {
    char buf[MAXSTRLEN];

    strcpy(rplName, curSpkr);
    strcat(rplName, "-");
    sprintf(buf, "%d", rplIdx);
    strcat(rplName, buf);
    strcat(rplName, RPLNMATSUFFIX);

    return rplName;
}

/* cz277 - xform */
char *MakeRplNVecName(char *curSpkr, int rplIdx, char *rplName) {
    char buf[MAXSTRLEN];

    strcpy(rplName, curSpkr);
    strcat(rplName, "-");
    sprintf(buf, "%d", rplIdx);
    strcat(rplName, buf);
    strcat(rplName, RPLNVECSUFFIX);

    return rplName;
}

/* cz277 - xform */
void ResetRplParts(ANNSet *annSet) {
    int i;
    RPLInfo *rplInfo;

    for (i = 1; i < MAXNUMRPLNMAT; ++i) {
        rplInfo = annSet->rplNMatInfo[i];
        if (rplInfo != NULL && rplInfo->inActive == TRUE) {
            if (rplInfo->rplNMat->matElems != rplInfo->saveRplNMatHost.matElems) {      /* speaker changed */
#ifdef CUDA
                SyncNMatrixDev2Host(rplInfo->rplNMat);  /* synchrnoise previous GPU updates to CPU */
#endif
                rplInfo->rplNMat->matElems = rplInfo->saveRplNMatHost.matElems;
#ifdef CUDA
                SyncNMatrixHost2Dev(rplInfo->rplNMat);  /* refresh GPU device*/
#endif
            }
        } 
    }
    for (i = 1; i < MAXNUMRPLNVEC; ++i) {
        rplInfo = annSet->rplNVecInfo[i];
        if (rplInfo != NULL && rplInfo->inActive == TRUE) {
            if (rplInfo->rplNVec->vecElems != rplInfo->saveRplNVecHost.vecElems) {      /* speaker changed */
#ifdef CUDA
                SyncNVectorDev2Host(rplInfo->rplNVec);  /* synchrnoise previous GPU updates to CPU */
#endif
                rplInfo->rplNVec->vecElems = rplInfo->saveRplNVecHost.vecElems;
#ifdef CUDA
                SyncNVectorHost2Dev(rplInfo->rplNVec);  /* refresh GPU device */
#endif
            }
        }   
    }
}

/* cz277 - pact */
int GetActFunNParmVector(LELink layerElem) {

    switch (layerElem->actFunInfo.actFunKind) {
    case AFFINEAF:
        return 2;
    case HERMITEAF:
        if (layerElem->actFunInfo.actParmVec[0] != NULL)
            return (-1) * ((int) layerElem->actFunInfo.actParmVec[0]->vecElems[0]);
        return ACTFUNPARMNOTLOAD;
    case LHUCRELUAF:
        return 1;
    case PRELUAF:
        return 1;
    case PARMRELUAF:
        return 2;
    case LHUCSIGMOIDAF:
        return 1;
    case PSIGMOIDAF:
        return 1;
    case PARMSIGMOIDAF:
        return 3;
    case LHUCSOFTRELUAF:
        return 1;
    case PSOFTRELUAF:
        return 1;
    case PARMSOFTRELUAF:
        return 3;
    default:
        return 0;
    }
}

/* cz277 - pact */
Boolean CacheActMatrixOrNot(ActFunKind actFunKind) {
    switch (actFunKind) {
    case PARMRELUAF:
    case PARMSIGMOIDAF:
    case PARMSOFTRELUAF:
        return TRUE;
    default:
        return FALSE;
    }
}



//cw564 - stimu -- begin
static StimuParam stp;

void ParsePhonePos(void) {
    FILE * file = fopen(stp.phonepos_fn, "r");
    if (!file) {
        HError(9999, "HNStimu: Phonepos file does not exist.");
    }
    
    fscanf(file, "%d %d", &(stp.num_phone), &(stp.dim_phonepos));
    stp.phonepos = CreateNMatrix(stp.heap, stp.num_phone, stp.dim_phonepos);
    
    NFloat mins[MAXARRAYLEN];
    NFloat maxs[MAXARRAYLEN];
    NFloat tmp;

    int i, j;
    for (i = 0; i < stp.num_phone; ++ i) {
        char * buf = malloc(sizeof(char) * MAXARRAYLEN);
        fscanf(file, "%s", buf);
        stp.phone_names[i] = buf;
        for (j = 0; j < stp.dim_phonepos; ++ j) {
            fscanf(file, "%f", &tmp);
            if (i == 0 || tmp < mins[j]) {
                mins[j] = tmp;
            }
            if (i == 0 || tmp > maxs[j]) {
                maxs[j] = tmp;
            }
            stp.phonepos->matElems[stp.dim_phonepos * i + j] = tmp;
        }
    }
    fclose(file);
    //normalization
    for (i = 0; i < stp.num_phone; ++ i) {
        for (j = 0; j < stp.dim_phonepos; ++ j) {
            tmp = stp.phonepos->matElems[stp.dim_phonepos * i + j];
            stp.phonepos->matElems[stp.dim_phonepos * i + j] = (tmp - mins[j]) / (maxs[j] - mins[j]);
        }
    }
    printf("%d %d\n", stp.num_phone, stp.dim_phonepos);
    for (i = 0; i < stp.num_phone; ++ i) {
        printf("%s ", stp.phone_names[i]);
        for (j = 0; j < stp.dim_phonepos; ++ j) {
            printf("%f ", stp.phonepos->matElems[stp.dim_phonepos * i + j]);
        }
        printf("\n");
    }
    //exit(0);
    
#ifdef CUDA
    SyncNMatrixHost2Dev(stp.phonepos);
#endif
}

static int nSTParm = 0;
static ConfParam * cSTParm[MAXGLOBS];
static StimuParam stp;

void InitStimu(MemHeap * heap) {
    stp.heap = heap;


    stp.lhuc_dist_var = 1.0;
    stp.lhuc_range_var = 0.01;
    stp.pos_dist = NULL;    
    
    stp.lhuc_regVal = NULL;
    stp.accumLHUCRegVal = 0;

    /*stp.oidx2phidx = CreateNVector(heap, 20000);//; (int *) malloc(sizeof(int) * 20000);
    int i;
    for (i = 0; i < 20000; ++ i) {
        stp.oidx2phidx->vecElems[i] = -1;
    }
    */
        
    stp.lhuc_penalty = 0.0;
    stp.lhuc_penalty_type = 0; 


    stp.accumKL = 0;

    stp.oidx2phidx = NULL;

    stp.phonepos_fn[0] = '\0';
    stp.stimu_penalty = 0;
    stp.num_phone = 0;
    stp.dim_phonepos = 0;
    stp.phonepos = NULL;
    stp.phoneidx_vec = NULL;
    stp.acti_surface = NULL;
    stp.grid_row = 0;
    stp.grid_col = 0;
    stp.grid_var = 1.0;

    stp.lhuc_surface = NULL;

    stp.visual_fn[0] == '\0';
    stp.visual_f = NULL;

    stp.stimu_dxFeaMat = NULL;

    stp.grid_var = 1.0;


    ResetStimu();

    //mixture -- begin
    stp.mixscaler = 0.5;
    stp.dnnscaler = 0.5;
    stp.mixnodes = 0;
    //mixture -- end

    int intVal, tmpInt;
    double doubleVal;
    Boolean boolVar;
    char buf[MAXSTRLEN], buf2[MAXSTRLEN];
    char *charPtr, *charPtr2;
    ConfParam *cpVal;

    nSTParm = GetConfig("HNSTIMU", TRUE, cSTParm, MAXGLOBS);

    if (nSTParm > 0) {
        if (GetConfStr(cSTParm, nSTParm, "PHONEPOS", buf)) {
            strcpy(stp.phonepos_fn, buf);
            //HError(9999, "%s\n", stp.phonepos_fn);
        }
        if (GetConfFlt(cSTParm, nSTParm, "LHUCPENALTY", &doubleVal)) {
            stp.lhuc_penalty = doubleVal;
        }
        if (GetConfFlt(cSTParm, nSTParm, "LHUCRANGEVAR", &doubleVal)) {
            stp.lhuc_range_var = doubleVal;
        }
        if (GetConfFlt(cSTParm, nSTParm, "LHUCDISTVAR", &doubleVal)) {
            stp.lhuc_dist_var = doubleVal;
        }
        if (GetConfInt(cSTParm, nSTParm, "LHUCPENALTYTYPE", &intVal)) {
            stp.lhuc_penalty_type = intVal;
        }
        if (GetConfFlt(cSTParm, nSTParm, "PENALTY", &doubleVal)) {
            stp.stimu_penalty = doubleVal;
        }
        if (GetConfStr(cSTParm, nSTParm, "VISUALFN", buf)) {
            strcpy(stp.visual_fn, buf);
        }
        if (GetConfFlt(cSTParm, nSTParm, "GRIDVAR", &doubleVal)) {
            stp.grid_var = doubleVal;
        }
        //mixture -- begin
        if (GetConfFlt(cSTParm, nSTParm, "MIXSCALER", &doubleVal)) {
            stp.mixscaler = doubleVal;
            stp.dnnscaler = 1 - doubleVal;
        }
        if (GetConfInt(cSTParm, nSTParm, "MIXNODES", &intVal)) {
            stp.mixnodes = intVal;
        }
        //mixture -- end
    }
    //HError(9999, "dist=%lf range=%lf\n", stp.lhuc_dist_var, stp.lhuc_range_var);
    if (!strcmp(stp.phonepos_fn, "")) {
        HError(9999, "HNStimu: Phonepos file is not assigned.");
    }
    ParsePhonePos();
}

StimuParam * STP(void) {
    return &stp;
}

int LabName2STPIdx(char * labname) {
    char * chr_ptr = NULL;
    if ((chr_ptr = strchr(labname, '[')) == NULL) {
        HError(9999, "LabName2STPIdx: Bad labname %s.", labname);
    }
    int triphone_name_len = (int)(chr_ptr - labname);
    char triphone_name[256];
    char mono_name[256];
    strncpy(triphone_name, labname, triphone_name_len);
    triphone_name[triphone_name_len] = '\0';
    if (!strcmp(triphone_name, "sil") || !strcmp(triphone_name, "sp")) {
        strcpy(triphone_name, "sil\0");
    }

    //printf("inittri=%s\n", triphone_name);
    
    char *left_ptr = strchr(triphone_name, '-'), *right_ptr = strchr(triphone_name, '+');
    if (left_ptr == NULL && right_ptr == NULL) {
        //printf("zhelizhelie?\n");
        strcpy(mono_name, triphone_name);
    }
    else {
        size_t mono_len = (size_t)(right_ptr - left_ptr) - 1;
        //printf("nimanim mono-len=%d rlen=%d, llen=%d\n", mono_len, (int)(right_ptr - triphone_name), (int)(left_ptr - triphone_name));
        strncpy(mono_name, left_ptr + 1, mono_len);
        mono_name[mono_len] = '\0';
    }
    //printf("%s %s\n", triphone_name, mono_name);
    int i;
    for (i = 0; i < stp.num_phone; ++ i) {
        if (!strcmp(stp.phone_names[i], mono_name)) {
            //HError(9999, "DBG, %s %d", mono_name, i);
            return i;
        }
    }
    HError(9999, "LabName2STPIdx: Bad mono-labname %s.", mono_name);
    return -1;
}


void PrintStimuKL(int num_samples) {
    printf("\t\tStimuKL=%lf\n", stp.accumKL / num_samples);
    stp.accumKL = 0;
}

void PrintLHUCRegVal(int num_samples, double ce, double lhuc_penalty) {
    printf("\t\tLHUCReg=%lf\n", stp.accumLHUCRegVal / num_samples);
    printf("\t\tCE+LHUCReg=%lf\n", (stp.accumLHUCRegVal * lhuc_penalty + ce) / num_samples);
    stp.accumLHUCRegVal = 0;
}

void ResetStimu() {
    stp.phonepos = NULL;
    stp.phoneidx_vec = NULL;
    stp.acti_surface = NULL;
    stp.lhuc_surface = NULL;
    stp.stimu_dxFeaMat = NULL;
    stp.weight_norms = NULL;
    stp.klvec = NULL;
    stp.sum_xFeaMat = NULL;
    stp.d_lhuc = NULL;
    stp.pos_dist = NULL;
    stp.lhuc_regVal = NULL;
    //mix
    int i;
    for (i = 0; i < 10; ++ i) {
        stp.comb_yFeaMat[i] = NULL;
        stp.comb_softmaxSum[i] = NULL;
    }
    stp.comb_dxFeaMat = NULL;
}

//cw564 - stimu -- end



