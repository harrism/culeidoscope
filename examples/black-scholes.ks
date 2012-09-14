extern printd(x);
extern printVector(vector x);
extern vector randVector(vector v range);
extern exp(x);
extern log(x);
extern sqrt(x);

def binary : 1 (x y) y;
def vector binary $ 1 (vector x vector y) y;

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
def bsCall(S X T)
  var sqrtT, d1, d2, CNDd1, CNDd2, expRT, R = 0.02, V = 0.3 in
    sqrtT = sqrt(T) :
    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT) :
    d2 = d1 - V * sqrtT :
    CNDd1 = CND(d1) :
    CNDd2 = CND(d2) :

    # Calculate Put
    expRT = exp(- R * T) :
    S * CNDd1 - X * expRT * CNDd2;

# S = Stock price, X = Option Strike, T = Option years, R = Riskless rate, V = Volatility rate
def bsPut(S X T)
  var sqrtT, d1, d2, CNDd1, CNDd2, expRT, R = 0.02, V = 0.3 in
    sqrtT = sqrt(T) :
    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT) :
    d2 = d1 - V * sqrtT :
    CNDd1 = CND(d1) :
    CNDd2 = CND(d2) :

    # Calculate Put
    expRT = exp(- R * T) :
    X * expRT * (1.0 - CNDd2) - S * (1.0 - CNDd1);
    
def vector black_scholes_call(N)
  var vector stockPrice[N], 
      vector optionStrike[N], 
      vector optionYears[N] in
    randVector(stockPrice, 30.0) $ 
    randVector(optionStrike, 100.0) $
    randVector(optionYears, 10.0) $
    map(bsCall, stockPrice, optionStrike, optionYears);

# single call (serial)
printd(bsCall(20, 10, 2));

# 100 calls (parallel)
printVector(black_scholes_call(100));



   
