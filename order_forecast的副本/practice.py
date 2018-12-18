import pandas as pd
import time
import datetime

dd = [1,2,3,4,5,7,8,9]

print(pd.Series(dd).rolling(window=3))

# s= dd[dd['a']==5,'c']

print()