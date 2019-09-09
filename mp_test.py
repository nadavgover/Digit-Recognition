import multiprocessing as mp
import numpy as np

def func(x):
    return x ** 2

if __name__ == '__main__':
    weights = np.array([1,2,3])
    weights1 = np.array([4,5,6])
    weightsi = [weights, weights1]
    with mp.Pool() as pool:
        results = pool.map(func, weightsi)

    results = np.array(results)

    print(results)