import pandas as pd
import urllib.request

URL = "http://wiski.tirol.gv.at/hydro/ogd/OGD_W.csv".
df = pd.read_csv(URL, encoding="ISO-8859-1")
print(df)


def fetch_level_data(url):
    df = pd.read_csv(url, encoding="ISO-8859-1", sep=';', parse_dates=['Zeitstempel in ISO8601'])
    df['timestamp_utc'] = df['Zeitstempel in ISO8601'].apply(lambda x: x.tz_convert('UTC')).values.astype('datetime64')
    df['date'] = df['timestamp_utc'].astype('datetime64[D]')
    return df


level_data = fetch_level_data(URL)
print(level_data['date'])
