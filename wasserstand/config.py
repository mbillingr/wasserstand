# Note: don't forget to set ~/.aws/credentials when using s3

DATA_SOURCE_URL = "https://wiski.tirol.gv.at/hydro/ogd/OGD_W.csv"

DATA_ROOT = "s3://kazemakase-data"

MODEL_ROOT = "s3://kazemakase-data/models"

DATAFILE_TAG = "wasser"

DATAFILE_RAW_TEMPLATE = f"{DATA_ROOT}/raw/%Y-%m-%d-%H:%M.parquet"

DATAFILE_TEMPLATE = f"{DATA_ROOT}/{DATAFILE_TAG}_%Y-%m-%d.parquet"
DATAFILE_ALL = f"{DATA_ROOT}/{DATAFILE_TAG}_*"
DATAFILE_LATEST = f"{DATA_ROOT}/latest_{DATAFILE_TAG}.parquet"

RESAMPLE_TO_MINUTES = [0, 15, 30, 45]
SAMPLE_INTERVAL_MINUTES = 15
