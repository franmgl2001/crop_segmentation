import os
import pickle
import numpy as np
import datetime
import geopandas as gpd


def get_doy(date):
    date = str(date)
    Y = date[:4]
    m = date[4:6]
    d = date[6:]
    date = "%s.%s.%s" % (Y, m, d)
    dt = datetime.datetime.strptime(date, "%Y.%m.%d")
    return dt.timetuple().tm_yday


meta_patch = meta_patch = gpd.read_file(os.path.join(".", "metadata.geojson"))


# Lock the 10110_15.pickle patch
patch_10110 = meta_patch[meta_patch["ID_PATCH"] == 10110]

patch_10110.reset_index(drop=True)
print(patch_10110)
dates = patch_10110["dates-S2"].iloc[0]

print(dates)

print(np.array([get_doy(d) for d in dates.values()]) + 60)


with open("../processed_pastis/pickles/10110_15.pickle", "rb") as f:
    data = pickle.load(f)

print(data["doy"])


### Conclusion the model takes the dates from January to december, now try to predict the model.
