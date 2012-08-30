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

#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Bitcode/ReaderWriter.h"
// llvm headers
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Value.h"


#include "llvm/Metadata.h"
#include <cstdio>
#include <stdlib.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
using namespace llvm;

extern Module *TheModule; 
extern IRBuilder<> Builder; 
extern std::map<std::string, AllocaInst*> NamedValues;

extern AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          const std::string &VarName);

static int lRunBitcodeVerifier(llvm::Module *fModule)
{
  llvm::PassManager *PM;
  int error = 0;
  PM = new llvm::PassManager();
  PM->add(llvm::createVerifierPass(llvm::ReturnStatusAction));
  if (PM->run(*fModule)) {
    error = 1;
  }
  delete PM;
  return error;
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


void CreateNVVMWrapperKernel(Module *M, Function *F, IRBuilder<> &Builder, std::string &kernelname) { 

  std::stringstream ss;
  ss << F->getName().data();
  ss << "_kernel";
  kernelname = ss.str();
  if (M->getFunction(kernelname)) {
    return;
  }
  
  const FunctionType *type = F->getFunctionType(); 
  std::vector<Type*> Params;

  Function *tidF = NULL; 

  tidF = M->getFunction("llvm.nvvm.read.ptx.sreg.tid.x"); 
  if (tidF==NULL) {
    // create an extern declaration for llvm-intrinsic
    Type *tidTy = Type::getInt32Ty(getGlobalContext()); 
    FunctionType *tidFunTy = FunctionType::get(tidTy,false);
    tidF = Function::Create(tidFunTy, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.tid.x", M);
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
    
    // Add arguments to variable symbol table.
    
    // Create an alloca for this variable.
    AllocaInst *Alloca = CreateEntryBlockAlloca(kerF, arg);

    // Store the initial value into the alloca.
    Builder.CreateStore(AI, Alloca);

    // Add arguments to variable symbol table.
    NamedValues[arg] = Alloca;
  }

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", kerF);
  Builder.SetInsertPoint(BB);
  std::vector<Value *> ArgsV;
  Value *tidreg = Builder.CreateCall(tidF, ArgsV, "calltmp");
  Idx = 0;
  std::vector<Value *> args; 
  for (Function::arg_iterator AI = kerF->arg_begin(); AI != kerF->arg_end();
       ++AI, ++Idx) {
    if (Idx < numParams) {
      std::vector<Value *>index;
      index.push_back(tidreg);
      Value *gep = Builder.CreateGEP(AI, index); 
      Value *loadInst = Builder.CreateLoad(gep);
      args.push_back(loadInst);
    } 
    else { 
      Value *result = Builder.CreateCall(F, args, "calltmp");
      std::vector<Value *>index;
      index.push_back(tidreg);
      Value *gep = Builder.CreateGEP(AI, index); 
      Value *storeInst = Builder.CreateStore(result, gep);
      break;
    } 
  }
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
} 

int
IRToBCFile(Module *M, std::string bcfile) { 
  if  (lRunBitcodeVerifier(M)) { 
    fprintf(stderr, "Verifier failed\n");
    return 1; 
  } 
  // write the bitcode to a file. 
  std::string error = ""; 
  raw_fd_ostream os(bcfile.c_str(), error); 
  //  if (error != "") { 
  //   fprintf(stderr, "failed to open bc file for writing\n");
  //}  
  WriteBitcodeToFile(M, os);
  return 0;
} 


int FunctionArity(char *name) { 
   std::string funcname = name;
   Function *F = TheModule->getFunction(funcname);
   
   if (F == NULL) { 
    return -1 ;
   }  
   return F->arg_size();
} ;


void RemoveIrrelevantFunctions(Module *M, std::string name)
{ 
  std::set<std::string> visited;
  std::vector<std::string> worklist;
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
     F->eraseFromParent();
  }
}

int 
CreatePtx(char *name, char *filename, std::string &kernelname) {
   std::string funcname = name;
   std::string ptxfilename = filename;
   Module *M = CloneModule(TheModule);

   RemoveIrrelevantFunctions(M,name);

   Function *F = M->getFunction(funcname);
   
   if (F == NULL) { 
    return -1 ;
   }  
   CreateNVVMWrapperKernel(M, F,Builder,kernelname);
   if (IRToBCFile(M, "./tmp.bc")) {
     fprintf(stderr, "ptx generation failed\n");
     return -1 ;
   }; 
   std::stringstream ss;
   char *nvvmcc=getenv("NVVMCC");
   if (nvvmcc == NULL) { 
     fprintf(stderr, "Set NVVMCC environment variable to point to right nvvmcc");
     return -1; 
   } 
   ss << nvvmcc; 
   ss << " ./tmp.bc -o ";
   ss << ptxfilename; 
   std::string nvvmcccmd = ss.str();
   int result = system(nvvmcccmd.c_str());
   if (result != 0) { 
     fprintf(stderr, "ptx generation failed\n");
     return -1;
   }
   return 0;
} 


