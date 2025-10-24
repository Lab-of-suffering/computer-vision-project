import numpy as np

if __name__ == '__main__':
    
    with open('output.txt', 'w') as f:
        npz = np.load("data/out/parameters.npz")
        f.write(f"keys: {npz.files}")
        for k in npz.files:
            a = npz[k]
            f.write(f"\n--- {k} ---\nshape: {a.shape}\ndtype: {a.dtype}\nvalues:\n{a}")