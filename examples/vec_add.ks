extern printVector(vector v);
extern randVector(vector v range);

def binary : 1 (x y) y;

def add(x y) x + y;

def vector vec_add(vector x vector y) map(add, x, y);

var vector x[257], vector y[257] in 
  randVector(x, 5) : 
  randVector(y, 2) :
  printVector(vec_add(x, y));


   
