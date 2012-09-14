extern printd(x);
extern printVector(vector x);
extern sqrt(x);

def printSqrt(x) printd(sqrt(x));

for i=1,i<10 in printSqrt(i*2);

def one(x) 1.0;

var vector a[10] in 
  printVector(map(sqrt, map(one, a)));


