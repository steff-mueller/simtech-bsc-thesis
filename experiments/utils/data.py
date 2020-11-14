from pathlib import Path
import json
import numpy as np
from models.integrators.base import TimeDataList

def td_x_to_hamiltonian(td_x_surrogate, model, mu):
    result = TimeDataList()
    for t,x in td_x_surrogate:
        result.append(t, model.Ham(x, mu))

    return result


def td_to_numpy(td: TimeDataList):  
    t = np.array(td.all_t())
    assert len(t.shape) == 1
    data = td.all_data_to_numpy() if hasattr(td, 'all_data_to_numpy') else np.array(td.all_data())

    if len(data.shape) == 1:
        return np.stack((t,data), axis=1)
    else:
        return np.concatenate((t.reshape(-1,1),data), axis=1)
    

def save_json(filepath, data):
    path = Path(*filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4))
