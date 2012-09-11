extern printVector(vector x);

def add(x y) x + y;

def one(x) 1.0;

def two(x) 2.0;

var vector a[10] in 
   printVector(map(add, map(two, a), map(one, a)));

   
