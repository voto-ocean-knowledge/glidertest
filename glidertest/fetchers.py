import pooch
import xarray as xr

server = "https://github.com/OceanGlidersCommunity/OG-format-user-manual/raw/main/og_format_examples_files"
data_source_og = pooch.create(
    path=pooch.os_cache("glidertest"),
    base_url=server,
    registry={
        "sea076_20230906T0852_R.nc": "sha256:0d8424ba021f151d8ed60aab8f7a799a251b29752d5521776b5d67b4b4f0212a",
    },
)


def load_sample_dataset(dataset_name="sea076_20230906T0852_R.nc"):
    if dataset_name in data_source_og.registry.keys():
        file_path = data_source_og.fetch(dataset_name)
        return xr.open_dataset(file_path)
    else:
        msg = f"Requested sample dataset {dataset_name} not known"
        raise ValueError(msg)
