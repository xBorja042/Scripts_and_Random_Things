import numpy as np
import pandas as pd
import pytz

import cryptocompare as cc
from forex_python.converter import CurrencyRates

import string

lentghs = ['2-3y', '5y', '1-5y', '5-10y',
          '10-15y', '15-20y', '20-25y', '10-20y', '10y', '20y', '30y', '20-40y']


geogs1 = ['SPANISH NAMES', 'SPANISH TESORO', 'ITALIAN GOVIES','SPANISH ILB','MADRID',
 'RATES','CENTRAL PORTUGAL','LA RIOJA','ATHENS','SPANISH SOVEREIGN BONDS', 'NETHERLANDS BV', 'NAVARRA', 'CANADA', 
 'CENTRAL SLOVAKIA', 'SPGB AUCTION', 'ARGENTINA','ASIA',
 'ITALY EASTERN', 'AUSTRIA',
 'GERMAN REGIONS', 'SPAIN PORTUGAL',
 'AUSTRALIAN','EUROPEAN FUND']
geogs = [pytz.country_names[key] + '_' + lentghs[np.random.randint(len(lentghs))] for key in pytz.country_names.keys()]
geogs = geogs + geogs1
geogs = geogs[0:30]

print(f" Total geophs ---> {len(geogs)}")

import pandas_datareader as pdr
nasdaq_values = [value for value in pdr.nasdaq_trader.get_nasdaq_symbols(retry_count=3, timeout=30, pause=None)['Security Name'].values.tolist()
                  if len(value) < 30]

nv = [value + '_' + lentghs[np.random.randint(len(lentghs))] for value in nasdaq_values[1000:1030]]

print(f" Total NasDaq Values ---> {len(nv)}")

# coin_list = cc.get_coin_list()
# coin_list.keys()

# krs = ['kr_' +  krypto for krypto in coin_list.keys()][200:230]

krs = [item for sublist in pd.read_csv('k.csv', sep=';').values.tolist() for item in sublist]
print(f" Total kryptos {len(krs)}")

c = CurrencyRates()

fx = ['fx_' + frx for frx in c.get_rates('USD').keys()]
print(f" Total Forex {len(fx)}")

def generate_products(l_p, veces = 10):
  l = list()
  veces = np.random.randint(veces)
  ls = string.ascii_lowercase
  client = ls[np.random.randint(0, 26)] + '_' + str(np.random.randint(0, 999))  + '_' + ls[np.random.randint(0, 26)].upper()
  times = list()
  for i in range(veces + 1):
    l.append(l_p[np.random.randint(len(l_p))])
    times.append(client)
  return l, times

lt_users, lt_products = list(), list()

for tipo in [fx, krs, nv, geogs]:
  if tipo != fx:
    for i in range(500):
      gen = generate_products(tipo)
      lt_users.append(gen[1])
      lt_products.append(gen[0])
  else:
    for i in range(100):
      gen = generate_products(tipo)
      lt_users.append(gen[1])
      lt_products.append(gen[0])

def generate_mixed_products(list_p, veces = 10):
  t = len(list_p)
  l = list()
  veces = np.random.randint(veces)
  ls = string.ascii_lowercase
  client = ls[np.random.randint(0, 26)] + '_' + str(np.random.randint(0, 999))  + '_' + ls[np.random.randint(0, 26)].upper()
  times = list()
  for product_type in list_p:
    for i in range(int((veces + 1)/t)):
      l.append(product_type[np.random.randint(len(product_type))])
      times.append(client)
  return l, times


for mixed_tipo in [[fx, krs], [nv, geogs], [geogs, nv, fx]]:
  for i in range(500):
    gen = generate_products(tipo)
    lt_users.append(gen[1])
    lt_products.append(gen[0])

t_users = [user for subl in lt_users for user in subl]
t_products = [product for subl in lt_products for product in subl]


v = [volume for volume in range(10000, 1000000, 300000)]
t_volumes = [v[np.random.randint(len(v))] for i in range(len(t_users))]

dft = pd.DataFrame({'client': t_users, 'volume': t_volumes, 'product': t_products,
                    'action': [{0:'BUY', 1: 'SELL'}[np.random.randint(2)] for i in range(len(t_products))]})


dict_prods = {}

for index, value in enumerate(dft['product'].unique()):
    nums, letters = '', ''
    for num in np.random.randint(0, 9, 10):
        nums += str(num)
    for letter in range(2):
        letters += string.ascii_lowercase[np.random.randint(0, 26)].upper()
    dict_prods[value] = letters + nums


dft['ISIN'] = dft['product'].map(dict_prods)

dft['isin_verb'] = dft['ISIN'] + '_' + dft['action']

to_mean = dft[['isin_verb','volume']]
to_mean = to_mean.groupby(['isin_verb'], as_index=False)['volume'].mean()
to_mean.columns = ['isin_verb','Mean']
 
data = pd.merge(dft,to_mean, on='isin_verb', how='left')
#Promedio
data['Mean']  = data['Mean'].apply(lambda x: int(x))
data['Promedio'] = data['volume']/data['Mean']


print(" Generated DataFrame ", dft.shape)

data.to_csv('synthetic_data.csv', sep = '|')