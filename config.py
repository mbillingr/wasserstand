# Note: don't forget to set ~/.aws/credentials when using s3

DATA_SOURCE_URL = "https://wiski.tirol.gv.at/hydro/ogd/OGD_W.csv"

DATA_ROOT = "s3://kazemakase-data"

DATAFILE_TAG = "wasser"

DATAFILE_TEMPLATE = f"{DATA_ROOT}/{DATAFILE_TAG}_%Y-%m-%d.parquet"
DATAFILE_ALL = f"{DATA_ROOT}/{DATAFILE_TAG}_*"
DATAFILE_LATEST = f"{DATA_ROOT}/latest_{DATAFILE_TAG}.parquet"
