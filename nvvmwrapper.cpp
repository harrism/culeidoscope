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

// llvm headers
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Value.h"


#include "llvm/Metadata.h"
#include <cstdio>
#include <stdlib.h>
#include <string>
#include <map>
#include <vector>
#include <deque>
#include <set>
#include <iostream>
#include <sstream>
#include "nvvm.h"
using namespace llvm;


extern Module *TheModule; 
extern std::map<std::string, AllocaInst*> NamedValues;

extern AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          const std::string &VarName,
                                          bool isVector);

static int lRunBitcodeVerifier(llvm::Module *fModule)
{
  llvm::PassManager *PM;
  int error = 0;
  PM = new llvm::PassManager();
  PM->add(llvm::createVerifierPass(llvm::PrintMessageAction));
  if (PM->run(*fModule)) {
    error = 1;
  }
  delete PM;
  return error;
}

void PruneUnrelatedFunctionsAndVariables(Module *M, std::string name)
{ 
  std::set<std::string> visited;
  std::deque<std::string> worklist;
  worklist.push_back(name);
  while (!worklist.empty()) { 
    std::string func = worklist.back(); 
    worklist.pop_back();
    visited.insert(func);
    Function *F = M->getFunction(func);      
    if (F == NULL) continue; 
    for (Function::iterator bi=F->begin(), be=F->end(); bi!=be; ++bi) {
      BasicBlock *bb = bi;
      for (BasicBlock::iterator ii=bb->begin(), ie=bb->end(); ii!=ie; ++ii) {
	      CallInst *call = dyn_cast<CallInst>(ii);
        if (call == NULL) 
          continue; 
        Function *called = call->getCalledFunction();
        if (called == NULL) continue;
	      std::string calledname = called->getName();
        if (visited.find(calledname) == visited.end()) { 
          worklist.push_back(calledname);
        } 
	    }
    }
  }

  // now walk the module and delete any unvisited functions.
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ) {
    Function *F = I++;
    if (visited.find(F->getName()) != visited.end()) 
      continue;
    if (!F->use_empty())
      worklist.push_back(F->getName());
    else
      F->eraseFromParent();
  }

  while (!worklist.empty()) {
    std::string func = worklist.front(); 
    worklist.pop_front();
    Function *F = M->getFunction(func);      
    if (F == NULL) continue; 
    if (!F->use_empty()) 
      worklist.push_back(func);
    else
      F->eraseFromParent();
  }

  // now erase unused global variables
  for (Module::global_iterator I = M->global_begin(); I != M->global_end(); ) {
    GlobalVariable *V = I++;
    V->eraseFromParent();
  }
}

// To be able to evaluate an expression f() on a GPU, 
// we create a wrapper function for F and mark it as 
// a kernel function with nvvm.annotations. 
// See the NVVM IR Specification document.   The data
// from host to device need to copied into separate
// memory, therefore the kernel function takes 
// pointer arguments.  The return value is written
// into the memory as well. 
//
// For f(x, y)
//
// kernel_f(*x, *y, *z) { 
//    t1 = x[tid];
//    t2 = x[tid];
//    t3 = f(t1,t2);
//    z[tid] = t4;
// } 
// Mark this with nvvm annotation as a kernel
// function. 

void CreateNVVMWrapperKernel(Module *M, Function *F, int N, IRBuilder<> &Builder, std::string &kernelname) { 

  PruneUnrelatedFunctionsAndVariables(M, F->getName());

  std::stringstream ss;
  ss << F->getName().data();
  ss << "_kernel";
  kernelname = ss.str();
  if (M->getFunction(kernelname)) {
    return;
  }
  
  const FunctionType *type = F->getFunctionType(); 
  std::vector<Type*> Params;

  Function *tidF = NULL, *ntidF = NULL, *ctaidF = NULL;

  tidF = M->getFunction("llvm.nvvm.read.ptx.sreg.tid.x"); 
  if (tidF==NULL) {
    // create an extern declaration for llvm-intrinsic
    Type *tidTy = Type::getInt32Ty(getGlobalContext()); 
    FunctionType *tidFunTy = FunctionType::get(tidTy,false);
    tidF = Function::Create(tidFunTy, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.tid.x", M);
  }

  ntidF = M->getFunction("llvm.nvvm.read.ptx.sreg.ntid.x"); 
  if (ntidF==NULL) {
    // create an extern declaration for llvm-intrinsic
    Type *ntidTy = Type::getInt32Ty(getGlobalContext()); 
    FunctionType *ntidFunTy = FunctionType::get(ntidTy,false);
    ntidF = Function::Create(ntidFunTy, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.ntid.x", M);
  }

  ctaidF = M->getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x"); 
  if (ctaidF==NULL) {
    // create an extern declaration for llvm-intrinsic
    Type *ctaidTy = Type::getInt32Ty(getGlobalContext()); 
    FunctionType *ctaidFunTy = FunctionType::get(ctaidTy,false);
    ctaidF = Function::Create(ctaidFunTy, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.ctaid.x", M);
  }

  // For each parameter in the original function
  // create a pointer to that param.
  unsigned numParams = type->getNumParams();
  for (unsigned i = 0; i < numParams; i++) { 
    Type *param_t = type->getParamType(i); 
    PointerType *p_t = PointerType::get(param_t, 0); 
    Params.push_back(p_t);
  } 
  Type *resultTy = F->getReturnType(); 
  PointerType *p_t = PointerType::get(resultTy, 0); 
  Params.push_back(p_t);
  Type *voidTy =  Type::getVoidTy(getGlobalContext());
  FunctionType *FT = FunctionType::get(voidTy,Params, false);
  Function *kerF = Function::Create(FT, Function::ExternalLinkage, kernelname, M);

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", kerF);
  Builder.SetInsertPoint(BB);

  // Calculate linear index from thread ID, CTA ID, and # threads per CTA
  std::vector<Value *> ArgsV;
  Value *tidreg = Builder.CreateCall(tidF, ArgsV, "calltmp");
  Value *ntidreg = Builder.CreateCall(ntidF, ArgsV, "calltmp");
  Value *ctaidreg = Builder.CreateCall(ctaidF, ArgsV, "calltmp");
  Value *idxreg = Builder.CreateMul(ntidreg, ctaidreg, "ntid_x_ctaid");
  idxreg = Builder.CreateAdd(idxreg, tidreg, "idx");

   // Create code to check if index > size, and if so, return 
  Value *CondV = Builder.CreateICmpULT(idxreg, 
                                       ConstantInt::get(IntegerType::getInt32Ty(getGlobalContext()), N),
                                       "ifcond");
        
  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(getGlobalContext(), "then", kerF);
  BasicBlock *ElseBB = BasicBlock::Create(getGlobalContext(), "else");

  Builder.CreateCondBr(CondV, ThenBB, ElseBB);
  
  // Emit then value -- load operands and call F
  Builder.SetInsertPoint(ThenBB);
  
  // Set names for all arguments.
  unsigned Idx = 0;
  for (Function::arg_iterator AI = kerF->arg_begin(); AI != kerF->arg_end();
       ++AI, ++Idx) {
    std::stringstream ss;
    ss << "arg";
    ss << Idx;
    std::string arg;
    ss >> arg; 
    AI->setName(arg);
    
    // Create an alloca for this variable.
    AllocaInst *Alloca = CreateEntryBlockAlloca(kerF, arg, false);

    // Add arguments to variable symbol table.
    NamedValues[arg] = Alloca;
  }

  Idx = 0;
  std::vector<Value *> args; 

  for (Function::arg_iterator AI = kerF->arg_begin(); AI != kerF->arg_end();
       ++AI, ++Idx) {
    if (Idx < numParams) {
      std::vector<Value *>index;
      index.push_back(idxreg);
      Value *gep = Builder.CreateGEP(AI, index); 
      Value *loadInst = Builder.CreateLoad(gep);
      args.push_back(loadInst);
    } 
    else { 
      Value *result = Builder.CreateCall(F, args, "calltmp");
      std::vector<Value *>index;
      index.push_back(idxreg);
      Value *gep = Builder.CreateGEP(AI, index); 
      Value *storeInst = Builder.CreateStore(result, gep);
      break;
    } 
  }

  Builder.CreateBr(ElseBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder.GetInsertBlock();
  
  // Emit else block.
  kerF->getBasicBlockList().push_back(ElseBB);
  Builder.SetInsertPoint(ElseBB);  
  
  Builder.CreateRetVoid();

  // Add the nvvm annotation that it is a kernel function. 
  LLVMContext &Context = getGlobalContext();
  Type *int32Type = Type::getInt32Ty(Context); 
  std::vector<Value *> Vals;
  NamedMDNode *nvvmannotate = M->getOrInsertNamedMetadata("nvvm.annotations");
  MDString *str = MDString::get(Context, "kernel");
  Value *one = ConstantInt::get(int32Type, 1);
  Vals.push_back(kerF);
  Vals.push_back(str);
  Vals.push_back(one);  
  MDNode *mdNode = MDNode::get(Context, Vals);

  nvvmannotate->addOperand(mdNode); 
  // kerF->dump();
} 



char *BitCodeToPtx(Module *M)
{
  M->dump();

  if  (lRunBitcodeVerifier(M)) { 
    fprintf(stderr, "Verifier failed\n");
    return 0; 
  } 

  // create memory buffer from LLVM Module
  std::vector<unsigned char> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
 
  Buffer.reserve(256*1024);
  WriteBitcodeToStream(M, Stream);

#define __NVVM_SAFE_CALL(X) do { \
    nvvmResult ResCode = (X); \
    if (ResCode != NVVM_SUCCESS) { \
      fprintf(stderr,"NVVM call (%s) failed. Error Code : %d", #X, ResCode); \
      exit(-1); \
    } \
  } while (0)
 
  // generate PTX
  nvvmCU CU;
  size_t Size;
  char *PtxBuf = 0;
 
  __NVVM_SAFE_CALL(nvvmCreateCU(&CU));
  const char *b = (const char *) &Buffer.front();
  __NVVM_SAFE_CALL(nvvmCUAddModule(CU, b, Buffer.size()));
 
  //const char *options = "-target=verify";
  nvvmResult result = nvvmCompileCU(CU, /*numOptions = */0, /*options = */0); //&options);
  if (result != NVVM_SUCCESS) {
    size_t logSize = 0;
    nvvmGetCompilationLogSize(CU, &logSize);
    char *log = new char[logSize];
    nvvmGetCompilationLog(CU, log);
    printf("NVVM Compilation Log:\n%s\n", log);
    delete [] log;
  }
  else {
    __NVVM_SAFE_CALL(nvvmGetCompiledResultSize(CU, &Size));
    PtxBuf = new char[Size+1];
    __NVVM_SAFE_CALL(nvvmGetCompiledResult(CU, PtxBuf));
    PtxBuf[Size] = 0; // Null terminate
    __NVVM_SAFE_CALL(nvvmDestroyCU(&CU));
  }
  
  return PtxBuf;
 
#undef __NVVM_SAFE_CALL
} // BitCodeToPtx



