
import numpy as np

def euler_formula(x, tol=1e-10):
    a = 0 if abs(np.cos(x)) < tol else np.cos(x)
    b = 0 if abs(np.sin(x)) < tol else np.sin(x)
    return a +b*1j

def FFT(P):
    n = len(P)
    if n == 1:
        return P
    w = euler_formula((2*np.pi)/n)
    Pe,Po= P[::2],P[1::2]
    ye,yo = FFT(Pe),FFT(Po)
    y = [0]*n
    for i in range(int(n/2)):
        y[i] = ye[i] + (w**i)*yo[i]
        y[i+int((n/2))] = ye[i] - (w**i)*yo[i]

    return y



def IFFT(P):
    def _IFFT(P):
        n = len(P)
        if n == 1:
            return P
        w = (euler_formula(2*np.pi/n)**(-1))
        Pe,Po = P[::2],P[1::2]
        ye,yo = _IFFT(Pe),_IFFT(Po)
        y = [0]*n
        for i in range(int(n/2)):
            y[i] = ye[i] + (w**i)*yo[i]
            y[i+int((n/2))] = ye[i] - (w**i)*yo[i]

        return np.array(y)
    return _IFFT(P)/len(P)

def polynomial_multiplication(f,g):
    if len(f) == len(g) and np.log2(len(f)).is_integer():
        resultant_polynomial =  np.array(IFFT(np.array(FFT(f + [0]*len(f)))*np.array(FFT(g + [0]*len(g)))))
        tol = 1e-12
        resultant_polynomial_clean = resultant_polynomial.copy()
        resultant_polynomial_clean.real[np.abs(resultant_polynomial.real) < tol] = 0
        resultant_polynomial_clean.imag[np.abs(resultant_polynomial_clean.imag) < tol] = 0
        return resultant_polynomial_clean

    else:
        return "The input polynomial dont have the same degree and its degree plus one is not a in the form of 2 power something"

print(polynomial_multiplication([-11,0,5,7],[-3,0,1,2]))