
# %% ---------------------------------------------------------------------------

import pandas as pd

hdf = pd.read_pickle("Hasbro.pickle")
husers = pd.Series([ user.get("screen_name") for user in hdf.user ])

sdf = pd.read_pickle("collectsideshow.pickle")
susers = pd.Series([ user.get("screen_name") for user in sdf.user ])

htdf = pd.read_pickle("hottoysofficial.pickle")
htusers = pd.Series([ user.get("screen_name") for user in htdf.user ])

all_users = husers.append(susers).append(htusers)
distr = all_users.value_counts()


# %% ---------------------------------------------------------------------------
