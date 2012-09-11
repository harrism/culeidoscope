extern printVector(vector x);
extern exp(x);
extern log(x);
extern sqrt(x);

def binary : 1 (x y) y;

def unary-(x) 0 - x;

def abs(x) if (x < 0) then -x else x;

def CND(d)
  var K, 
      cnd,
      A1 = 0.31938153,
      A2 = -0.356563782,
      A3 = 1.781477937,
      A4 = -1.821255978,
      A5 = 1.330274429,
      RSQRT2PI = 0.39894228040143267793994605993438 in   
    K = 1.0 / (1.0 + 0.2316419 * abs(d)) :    
    cnd = RSQRT2PI * exp(- 0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))) :
    if (d > 0) then 1.0 - cnd else cnd;

# S = Stock price, X = Option Strike, T = Option years, R = Riskless rate, V = Volatility rate    
def BlackScholesCall(S X T R V)
  var sqrtT, d1, d2, CNDd1, CNDd2, expRT in
    sqrtT = sqrt(T) :
    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT) :
    d2 = d1 - V * sqrtT :
    CNDd1 = CND(d1) :
    CNDd2 = CND(d2) :

    # Calculate Call
    expRT = exp(- R * T) :
    S * CNDd1 - X * expRT * CNDd2;

# S = Stock price, X = Option Strike, T = Option years, R = Riskless rate, V = Volatility rate
def BlackScholesPut(S X T R V)
  var sqrtT, d1, d2, CNDd1, CNDd2, expRT in
    sqrtT = sqrt(T) :
    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT) :
    d2 = d1 - V * sqrtT :
    CNDd1 = CND(d1) :
    CNDd2 = CND(d2) :

    # Calculate Put
    expRT = exp(- R * T) :
    X * expRT * (1.0 - CNDd2) - S * (1.0 - CNDd1);
    

def add(x y) x + y;

def one(x) 1.0;

def two(x) 2.0;

var vector a[10] in 
   printVector(map(add, map(two, a), map(one, a)));

   
