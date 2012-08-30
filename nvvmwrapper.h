#include <string>
extern int CreatePtx(char *name, char *filename, std::string &kernel );
extern int FunctionArity(char *name);
extern void LaunchOnGpu(const char *kernel, 
                 unsigned funcarity, 
                 unsigned N, 
                 void **args, 
                 void *resbuf,
                 const char *filename) ;
