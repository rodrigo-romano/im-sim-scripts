import numpy as np
import pickle

data = np.load("SHAcO_qp_rhoP1e-3_kIp5.pkl",allow_pickle=True)

def browse_dict(d,o):
    for key in d:
        print(f"{o}: <{key}>")
        if isinstance(d[key],dict):
            browse_dict(d[key],o+1)
        elif isinstance(d[key],np.ndarray):
            buf = d[key].copy()
            print(f" size: [{buf.shape}], type: {buf.dtype}")
            d[key] = buf.ravel().tolist()
        elif isinstance(d[key],list):
            n = len(d[key])
            print(f" size: [{n}].type: {type(d[key][0])}")
            for k in range(n):
                if isinstance(d[key][k],np.ndarray):
                    buf = d[key][k].copy()
                    print(f" #{k}. size: [{buf.shape}], type: {buf.dtype}")
                    d[key][k] = buf.ravel().tolist()
        else:
            print(f" type: {type(d[key])}")


browse_dict(data,0)

with open("SHAcO_qp_rhoP1e-3_kIp5.rs.pkl","wb") as f:
    pickle.dump(data,f)