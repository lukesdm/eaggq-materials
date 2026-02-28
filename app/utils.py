from dataclasses import dataclass
from enum import Enum
from typing import Literal

import planetary_computer
import pystac_client
import shapely


@dataclass
class CBFilterParams:
    max_cloud: float = 100.0
    min_veg: float = 0.0
    max_veg: float = 100.0
    min_soil: float = 0.0
    max_soil: float = 100.0


@dataclass
class ContentBasedSearchParams(CBFilterParams):
    satellite: str = None
    stac_items: list = None
    aoi: str | dict = None  # WKT string or dictionary representing a GeoJSON geometry
    cbs_resolution: int = (
        None  # Override ground sampling distance for downsampling (target CRS units)
    )


class SCL(Enum):
    """Classes in the Scene Classification Layer"""

    Mask = 0
    Sat = 1
    Dark = 2
    Shadow = 3
    Veg = 4
    Bare = 5
    Water = 6
    Unknown = 7
    Cl_Med = 8
    Cl_High = 9
    Cl_Cirrus = 10
    Snow = 11


scl_colors = {
    SCL.Mask: "#000000",
    SCL.Sat: "#FF0000",
    SCL.Dark: "#3F3F3F",
    SCL.Shadow: "#833C09",
    SCL.Veg: "#00FF00",
    SCL.Bare: "#FFFF03",
    SCL.Water: "#0300CC",
    SCL.Unknown: "#757171",
    SCL.Cl_Med: "#AEAAAA",
    SCL.Cl_High: "#D0CECE",
    SCL.Cl_Cirrus: "#00CCFF",
    SCL.Snow: "#FF66FF",
}


def drop_old_dupes(stac_items):
    """
    Sentinel-2 results from MSPC can have multiple items for the same acquisition -
    this is mainly due to baseline updates. So, filter out the ones with the older generation time.
    Assume results are already sorted by generation time (ascending).
    """
    to_keep = {}
    for item in stac_items:
        # Strip generation time from granule id.
        product_id = item.id[:-16]
        to_keep[product_id] = item
    return list(to_keep.values())


def find_stac_items(
    aoi: shapely.Polygon, time_period: str, method: Literal["intersects", "contains"]
):
    """Performs a STAC search for the given AoI and time period, using the Microsoft Planetary Computer STAC API."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search_poly_coords = list(aoi.exterior.coords)

    cql2_contains = {
        "op": "s_contains",
        "args": [
            {"property": "geometry"},
            {"type": "Polygon", "coordinates": [search_poly_coords]},
        ],
    }

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        datetime=time_period,
        filter=cql2_contains if method == "contains" else None,
        intersects=aoi if method == "intersects" else None,
        sortby="s2:generated_time",
    )
    items = search.item_collection()

    return drop_old_dupes(items)
