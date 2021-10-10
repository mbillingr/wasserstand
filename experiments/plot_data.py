import dask.dataframe
import matplotlib.pyplot as plt

import config

water_data = dask.dataframe.read_parquet(config.DATA_ROOT + "*")


def plot_station(name):
    station_data = water_data[water_data["Stationsname"] == name]

    x = station_data["timestamp_utc"].compute()
    y = station_data["Wert"].compute()

    plt.plot(x, y - y.mean(), label=name)


for station in sorted(water_data["Stationsname"].unique()):
    print(station)

plot_station("Zirl")
plot_station("Innsbruck")

plt.legend()
plt.show()
