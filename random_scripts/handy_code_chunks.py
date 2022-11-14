
def clean_dataset(df):
    "This function cleans rows fulls of nans. Found in stack overflow."
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

## SNIPETS

## CALLBACKS

class reduce_prints(tf.keras.callbacks.Callback):
    # Print the loss and mean absolute error after each epoch
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 ==0:
            print('Epoch {}: Average loss is {:7.2f}, mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['val_loss']))



## PITIDINNN
import winsound

def alarm():
    frequency = 800  # Set Frequency To 2500 Hertz
    duration = 600  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    frequency = 600  # Set Frequency To 2500 Hertz
    duration = 600  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    frequency = 800  # Set Frequency To 2500 Hertz
    duration = 600  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    frequency = 600  # Set Frequency To 2500 Hertz
    duration = 600  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

alarm()




## MEASURE TIME

import time
from datetime import timedelta
epoch_logger = EpochLogger()
start = time.time()
elapsed = (time.time() - start)

str(timedelta(seconds=elapsed))[:-7]




## PLOTEADO DINAMICO EN JUPYTER 

%matplotlib inline
import matplotlib.pyplot as plt
import pylab as pl
import time
from IPython import display

df_bbg['n_words'] = df_bbg['Content'].apply(lambda x: len(str(x).split(' ')))

for i in [(100, 40), (60, 40), (50, 20), (20,10), (10,10)]:
    plt.hist(df_bbg['n_words'], bins = i[1], range=[0,i[0]])
    plt.legend(['Message Lenght'])
    plt.show()
    print(" Calculating ")
    time.sleep(3)
    display.clear_output(wait=True)
    display.display(pl.gcf())


## DISPLAY ALL PANDAS DATA
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



## JOBLIB AND THREADS
import threading
from joblib import Parallel, delayed
 
l2 = [1,2,3,4] * 20
def mult(inte):
    print(threading.get_ident())
    return [inte**2]
 
embs = Parallel(n_jobs=5, prefer="threads")(delayed(mult)(elem) for elem in l2)
embs




## PARA BORRAR UN ENV
conda deactivate
conda env remove --name ENV
Posibilidad de tener que borrar el dir.


## PARA CREAR UN ENV
conda create -p /opt/xva/devel/ENV
(desde yaml)
conda env create --name envname --file=algo.yml
conda env create -p /opt/xva/devel/bert_embs --file=bert.yml

## PARA LISTAR ENVS
conda env list
## PARA LISTAR PACKAGES
!pip list --local

## PARA INSTALAR PACKAGES / MOSTRAR EL ENVIRONMENT EN EL LAUNCHER
$ source activate /opt/xva/devel/bert_embs/
(cenv)$ conda install ipykernel
(cenv)$ ipython kernel install --user --name=<any_name_for_kernel>
(cenv($ conda deactivate



ARREGLAR EL ERROR 500
ESTA PETANDO EL CLUSTER POR ERROR 500. PODRIA SER UN PROBLEMA DE HUGGING FACE AL CARGAR MODELOS.
DESDE HOME:
cd ./.cache/
du -h
rm -r DIRE_QUE_MAS_PESE (hugggin)

PROBLEMA DE CONDA
LISTAR PAQUETES ENVIRONTMET
ls /xx/YY/lib/python3.7/site-packages/
FULMIAR EL PAQUETE FEO
rm -R /XX/YY


SPARK
descachear
spark.catalog.clearCache()

spark.createDataFrame(data = (("uno", 1), ("dos", 2), ("tres", 3)), schema=["cuerda", "entero"])
