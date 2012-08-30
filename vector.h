struct DVector {
  double  *ptr;      
  int     length;
};

/*extern void dvec(dvector *, double, double);
extern double printd(dvector);
extern double print(double);
extern double vlen(dvector);
extern double vlen1(double size, void *ptr);*/
extern void vector_map(char *, DVector *, DVector *);
