#%%
import pandas as pd
from datetime import date
from nsepy import get_history
nifty_opt = get_history(symbol="NIFTY",
                        start=date(2015,1,1),
                        end=date(2015,1,10),
                        index=True,
                        option_type='CE',
                        strike_price=8200,
                        expiry_date=date(2015,1,29))
# %%
