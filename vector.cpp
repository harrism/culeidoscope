#include <stdio.h>
#include "vector.h"
#include "malloc.h"
#include "nvvmwrapper.h"
#include <string> 

/*void dvec(dvector *vp, double dsize , double elem) 
{
  int s = (int) (sizeof(double)*dsize);
  double *p = (double *)malloc(s) ;
  unsigned i = 0;
  for (i = 0; i < dsize; i++)
       p[i] = elem;
  vp->vbase = p;
  vp->size = dsize;
}


double
printd(dvector v) { 
  unsigned i = 0;
  unsigned size = v.size;
  printf("[ ");
  double *p = (double *)v.vbase;
  for (i = 0; i < size; i++)
    printf("%f ",p[i]);
  printf("]\n");
  return size;
} 

double
print(double f) {
  printf("%f\n",f);
  return 1;
} 
double vlen(dvector v) { 
  return v.size;
} */

void 
vector_map(char *name, DVector *res, DVector *args) { 
  char *filename = "eval.ptx";
  unsigned arity = FunctionArity(name);
  
  std::string kernel; 
  if (CreatePtx(name, filename,kernel)) { 
    fprintf(stderr, "Unable to create ptx file\n");
  } 
  
  void **argsbuf = (void **) malloc(sizeof(void *)*arity);
  unsigned pos;
  
  for (pos = 0; pos < arity; pos++) 
    argsbuf[pos] = args[pos].ptr;
  
  res->length= args[0].length;
  res->ptr = (double *) malloc(res->length * sizeof(double));  
  
  if (res->ptr == NULL) { 
     fprintf(stderr,"Could not allocate host memory\n" );
     return ;
  } 
  
  LaunchOnGpu(kernel.c_str(), arity, res->length, argsbuf, res->ptr, "eval.ptx");
} 


