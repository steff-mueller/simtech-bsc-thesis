from pathlib import Path
import json
import numpy as np
from models.dyn_sys import TimeDataList

def td_x_to_hamiltonian(td_x_surrogate, model, mu):
    result = TimeDataList()
    for t,x in td_x_surrogate:
        result._t.append(t)
        result._data.append(model.Ham(x, mu))

    return result

def time_data_list_to_numpy(td: TimeDataList):
    return np.array((td._t, td._data)).transpose()

def save_json(filepath, data):
    path = Path(*filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4))
