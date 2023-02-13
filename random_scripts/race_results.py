import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def hour_to_seconds(time) -> int:
    seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], str(time).split(":")))
    return seconds

def compute_q(s: int, lqs: list) -> int:
    q = 0
    if s < lqs[0]:
        q = 1
    elif s >= lqs[0] and s < lqs[1]:
        q = 2
    elif s >= lqs[1] and s < lqs[2]:
        q = 3
    else:
        q = 4
    return q
        

def compute_t_diff(s: int, quart: int, lqs: list):
    try:
        dt = s - lqs[quart-2]
    except:
        dt = 0
    dt = str(datetime.timedelta(seconds=dt))[:-7]
    return dt
    
    


dfr = pd.read_excel("race_results.xlsx")
dfr["nombre"] = dfr["Nombre"].apply(lambda x: x.lower())
dfr = dfr.loc[~pd.isnull(dfr["Ritmo"])]
               
participants = ["jimenez coelho juan josé", "fernández garcía manuel",
                "toimil moya julio", "rodríguez rodríguez francisco javier"]

use_cols = ['Dorsal', 'Nombre', 'Club', 'Tiempo',
       'Ritmo', 'nombre']
dfr = dfr[use_cols]


dfr["time_seconds"] = dfr["Tiempo"].apply(hour_to_seconds)
qs = np.percentile(dfr["time_seconds"], [25, 50, 75, 100])
dfr["quartile"] = dfr["time_seconds"].apply(lambda x: compute_q(x, qs))
dfr["time_diff"] = dfr.apply(lambda x: compute_t_diff(x["time_seconds"], x["quartile"], qs), axis=1)


dfp = dfr.loc[dfr["nombre"].isin(participants)]
dff = dfp[['Dorsal', 'Nombre', 'Club', 'Tiempo', 'Ritmo', 'nombre', 
       'quartile', 'time_diff']]

graph = sns.kdeplot(data=dfr["Ritmo"], linewidth=4)
h0 = 0.03

for key, value in enumerate(dff["Ritmo"].to_numpy()):
    graph.axvline(value, linewidth=0.7)
    plt.text(dff["Ritmo"].to_numpy()[key], h0, dff["Nombre"].to_numpy()[key][:10], horizontalalignment='left', size='medium', color='black', weight='semibold')
    h0 += 0.02

plt.show()