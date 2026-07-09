import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # For the paper 'Efficient semantic content-based image retrieval with cloud-optimized raster formats'
    # Authors: Luke McQuade, Martin Sudmanns, Dirk Tiede
    # February 2026

    import marimo as mo

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import planetary_computer
    import pystac_client
    from rasterio.enums import Resampling
    import rioxarray
    import shapely
    import stackstac
    import xarray


    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray

    import time

    from utils import SCL, scl_colors

    hv.extension("bokeh")
    return (
        Resampling,
        SCL,
        gpd,
        mo,
        np,
        pd,
        planetary_computer,
        pystac_client,
        scl_colors,
        shapely,
        stackstac,
        xarray,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Effects of spatial distribution on class distribution when downsampling

    This notebook gives a demonstration of the effect of how the spatial distribution of geographical features (as they appear in the [Sentinel-2 L2A Scene Classification Layer](https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-L2AAlgorithmsS2-Processing-L2A-Algorithmstrue) (SCL)) affects class distribution when downsampled. The SCLs for various scenes are randomly spatially shuffled, and downsampled. The class distributions of these are compared against downsampled versions of the original.
    """)
    return


@app.cell
def _(gpd, shapely):
    aois = {
        "stockerau_full_8000_0": gpd.GeoDataFrame(
            geometry=[
                shapely.from_wkt(
                    "POLYGON ((584214 5351624, 584214 5359624, 576214 5359624, 576214 5351624, 584214 5351624))"
                )
            ],
            crs=32633,
        ),
        "rdd_extent_8000_0": gpd.GeoDataFrame(
            geometry=[
                shapely.from_wkt(
                    "POLYGON ((399767 4601400, 399767 4609400, 391767 4609400, 391767 4601400, 399767 4601400))"
                )
            ],
            crs=32630,
        ),
    }
    return (aois,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Load data

    Perform a STAC lookup for a set of images, then load the SCL data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### STAC lookup
    """)
    return


@app.cell
def _(planetary_computer, pystac_client):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Define selected STAC IDs, and their associated AoI names
    selected_items = {
        "S2B_MSIL2A_20230824T110619_R137_T30TUM_20230824T201347": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230814T110629_R137_T30TUM_20230814T182401": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230804T110619_R137_T30TUM_20230804T151800": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230725T110629_R137_T30TUM_20230725T160904": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230715T110629_R137_T30TUM_20230715T151518": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230705T110629_R137_T30TUM_20230705T163129": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230625T110619_R137_T30TUM_20230625T164707": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230615T110629_R137_T30TUM_20230615T205813": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230605T110619_R137_T30TUM_20230605T163049": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230526T110629_R137_T30TUM_20230526T153324": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230516T110619_R137_T30TUM_20230516T151848": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230506T110619_R137_T30TUM_20230506T152322": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230829T110621_R137_T30TUM_20230829T181513": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230819T110621_R137_T30TUM_20230819T204044": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230809T110621_R137_T30TUM_20230809T182051": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230730T110621_R137_T30TUM_20230730T182952": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230720T110621_R137_T30TUM_20230720T191334": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230710T110621_R137_T30TUM_20230710T193234": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230630T110621_R137_T30TUM_20230630T203024": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230620T110621_R137_T30TUM_20230620T202552": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230610T110621_R137_T30TUM_20230610T202756": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230531T110621_R137_T30TUM_20230531T203329": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230521T110621_R137_T30TUM_20230521T182158": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230511T110621_R137_T30TUM_20230511T182453": "rdd_extent_8000_0",
        "S2A_MSIL2A_20230501T110621_R137_T30TUM_20230711T225224": "rdd_extent_8000_0",
        "S2B_MSIL2A_20230830T094549_R079_T33UWP_20230830T171310": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230820T094549_R079_T33UWP_20230820T162026": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230810T094549_R079_T33UWP_20230810T142216": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230731T094549_R079_T33UWP_20230731T141335": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230714T095559_R122_T33UWP_20230714T152006": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230704T095559_R122_T33UWP_20230704T142637": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230624T095559_R122_T33UWP_20230624T142946": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230614T095559_R122_T33UWP_20230614T142248": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230604T095559_R122_T33UWP_20230604T174403": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230525T095559_R122_T33UWP_20230525T141502": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230515T095559_R122_T33UWP_20230515T150740": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230505T100029_R122_T33UWP_20230505T204335": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230828T100031_R122_T33UWP_20230828T171557": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230818T100031_R122_T33UWP_20230818T174140": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230808T100031_R122_T33UWP_20230808T181715": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230729T100031_R122_T33UWP_20230729T150804": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230719T100031_R122_T33UWP_20230719T154341": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230709T100031_R122_T33UWP_20230709T154425": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230629T100031_R122_T33UWP_20230629T191411": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230619T100031_R122_T33UWP_20230619T172742": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230609T100031_R122_T33UWP_20230609T173715": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230530T100031_R122_T33UWP_20230530T153335": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230520T100031_R122_T33UWP_20230520T155339": "stockerau_full_8000_0",
        "S2A_MSIL2A_20230510T100031_R122_T33UWP_20230510T154651": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230621T094549_R079_T33UWP_20230621T143955": "stockerau_full_8000_0",
        "S2B_MSIL2A_20230701T094549_R079_T33UWP_20230701T152448": "stockerau_full_8000_0",
    }

    _search = catalog.search(
        collections=["sentinel-2-l2a"], ids=selected_items.keys()
    )
    _items = (
        _search.item_collection()
    )  # Note, not returned in same order as above.
    items = {item.id: item for item in _items}
    return items, selected_items


@app.cell
def _(pd, selected_items):
    item_aois = pd.DataFrame(
        {"id": selected_items.keys(), "aoi": selected_items.values()}
    ).set_index("id")
    return (item_aois,)


@app.cell
def _(pd, selected_items):
    item_dates = pd.DataFrame(
        {
            "id": selected_items.keys(),
            "date_str": [
                f"{s[11:15]}-{s[15:17]}-{s[17:19]}" for s in selected_items.keys()
            ],
        }
    ).set_index("id")
    return (item_dates,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Load SCL data

    At native (20m) and downsampled resolutions.
    """)
    return


@app.cell
def _(stackstac):
    def load_scl(stac_item, aoi_gdf, resolution):
        if stac_item.properties.get("proj:code") is not None:
            epsg = int(stac_item.properties["proj:code"][5:])
        elif stac_item.properties.get("proj:epsg") is not None:
            epsg = stac_item.properties["proj:epsg"]
        else:
            raise Exception("Couldn't determine EPSG code from STAC item.")

        assert epsg == aoi_gdf.crs.to_epsg(), (
            f"STAC item EPSG = {epsg}, AoI EPSG = {aoi_gdf.crs.to_epsg()}. They should match."
        )

        stack = stackstac.stack(
            stac_item,
            assets=["SCL"],
            epsg=epsg,
            resolution=resolution,
            bounds_latlon=aoi_gdf.to_crs(4326).total_bounds,
            dtype=int,
            fill_value=0,
        )
        # Clip to the actual AoI geometry, rather than just the bounding rectangle.
        stack = stack.rio.clip(aoi_gdf.geometry)
        return stack.squeeze().compute()

    return (load_scl,)


@app.cell
def _(aois, items, load_scl, selected_items):
    scl_20m = {
        item_id: load_scl(
            items[item_id], aoi_gdf=aois[selected_items[item_id]], resolution=20
        )
        for item_id in selected_items
    }
    return (scl_20m,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Downsample the SCLs
    """)
    return


@app.cell
def _(Resampling):
    def downsample_2d(da, factor):
        assert da.rio.crs is not None, "CRS is required for downsampling."
        new_width = int(da.rio.width / factor)
        new_height = int(da.rio.height / factor)

        return da.rio.reproject(
            da.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.nearest,
        )

    return (downsample_2d,)


@app.cell
def _(downsample_2d, scl_20m, selected_items):
    scl_100m = {
        item_id: downsample_2d(scl_20m[item_id], factor=100 / 20)
        for item_id in selected_items
    }

    scl_200m = {
        item_id: downsample_2d(scl_20m[item_id], factor=200 / 20)
        for item_id in selected_items
    }
    return scl_100m, scl_200m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spatially shuffle the SCLs

    And downsample these.
    """)
    return


@app.cell
def _(np):
    RANDOM_SEED = 20250804115106
    ITERATIONS = 10

    rng = np.random.default_rng(seed=RANDOM_SEED)
    return ITERATIONS, rng


@app.cell
def _(rng, xarray):
    def shuffle_da(da):
        original_shape = da.shape
        flat_values = da.values.flatten()
        rng.shuffle(flat_values)
        da_shuffled = xarray.DataArray(
            flat_values.reshape(original_shape),
            dims=da.dims,
            coords=da.coords,
            attrs=da.attrs
        )
        return da_shuffled

    return (shuffle_da,)


@app.cell
def _(ITERATIONS, scl_20m, selected_items, shuffle_da):
    scl_shuffled_20m = {
        item_id: [shuffle_da(scl_20m[item_id]) for i in range(ITERATIONS)]
        for item_id in selected_items
    }
    return (scl_shuffled_20m,)


@app.cell
def _(downsample_2d, scl_shuffled_20m, selected_items):
    scl_shuffled_100m = {
        item_id: [
            downsample_2d(da, factor=100 / 20) for da in scl_shuffled_20m[item_id]
        ]
        for item_id in selected_items
    }

    scl_shuffled_200m = {
        item_id: [
            downsample_2d(da, factor=200 / 20) for da in scl_shuffled_20m[item_id]
        ]
        for item_id in selected_items
    }
    return scl_shuffled_100m, scl_shuffled_200m


@app.cell
def _(mo):
    mo.md(r"""
    ## Calculate and compare the class distributions

    Compare all against the original SCLs at native (20m) resolution.
    """)
    return


@app.cell
def _(ITERATIONS, SCL, np, pd):
    # The analysis here assumes images and AoIs have been selected such that there aren't any NoData values.
    # So the versions of these functions don't have specific handling for excluding NoData from the distributions.


    def count_cats(arr, n_cats=12):
        cat_bins = np.arange(0, n_cats + 1)
        counts = np.histogram(arr.values, bins=cat_bins)[0]
        df = pd.DataFrame(
            {
                "category": cat_bins[:-1],
                "count": counts,
                "pc": 100.0 * counts / arr.size,
            }
        )
        return df


    def calc_dist(da):
        df = count_cats(da)
        id = str(da["id"].values)
        df["id"] = id
        df = df.pivot(index="id", columns="category", values="pc")
        df.columns = [item.name for item in SCL]
        df["Total_pixels"] = da.size
        return df


    def calc_dists(das):
        dists = []
        for da in das:
            df = calc_dist(da)
            dists.append(df)
        return pd.concat(dists)


    def calc_dists_2(da_dict):
        dists = []
        for item_id, das in da_dict.items():
            for i in range(ITERATIONS):
                df = calc_dist(das[i])
                df["iteration"] = i
                df = df.set_index("iteration", append=True)
                dists.append(df)
        return pd.concat(dists)

    return calc_dists, calc_dists_2


@app.cell
def _(calc_dists, scl_100m, scl_200m, scl_20m):
    scl_20m_dists = calc_dists(scl_20m.values())

    unshuffled_100m_dists = calc_dists(scl_100m.values())                                 

    unshuffled_200m_dists = calc_dists(scl_200m.values())

    unshuffled_100m_mce = (scl_20m_dists - unshuffled_100m_dists).drop(columns=["Total_pixels"]).abs().max(axis=1).sort_index()

    unshuffled_200m_mce = (scl_20m_dists - unshuffled_200m_dists).drop(columns=["Total_pixels"]).abs().max(axis=1).sort_index()
    return (
        scl_20m_dists,
        unshuffled_100m_dists,
        unshuffled_100m_mce,
        unshuffled_200m_dists,
        unshuffled_200m_mce,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Downsampled at 100m × 100m resolution
    """)
    return


@app.cell
def _(calc_dists_2, pd, scl_20m_dists, scl_shuffled_100m):
    shuffled_100m_dists = calc_dists_2(scl_shuffled_100m)
    shuffled_100m_mce = (
        (scl_20m_dists - shuffled_100m_dists)
        .drop(columns=["Total_pixels"])
        .abs()
        .max(axis=1)
    )

    _grp = shuffled_100m_mce.groupby("id")
    shuffled_100m_mce_stats = pd.DataFrame(
        {
            "mean": _grp.mean(),
            "std": _grp.std(),
            "min": _grp.min(),
            "max": _grp.max(),
            "median": _grp.median()
        }
    )
    shuffled_100m_mce_stats
    return shuffled_100m_dists, shuffled_100m_mce, shuffled_100m_mce_stats


@app.cell
def _(item_aois, item_dates, pd, shuffled_100m_mce, unshuffled_100m_mce):
    SUBPLOT_WIDTH=400

    def plot_mce(df_shuffled_mce, df_unshuffled_mce):
        _df_shf = df_shuffled_mce.join(item_dates).sort_values(by="date_str")
        _df_unshf = df_unshuffled_mce.join(item_dates).sort_values(by="date_str")
        return (
            _df_shf.hvplot.box(by="id")
            * _df_unshf.hvplot.scatter(color="red")
        ).opts(show_legend=False, width=SUBPLOT_WIDTH)


    def where_aoi(df, aoi):
        return df[df["aoi"] == aoi]


    _df_shuffled_mce = pd.DataFrame({"mce": shuffled_100m_mce}).join(item_aois)
    _df_unshuffled_mce = pd.DataFrame({"mce": unshuffled_100m_mce}).join(item_aois)

    _df_shuffled_mce_1 = where_aoi(_df_shuffled_mce, "rdd_extent_8000_0")
    _df_shuffled_mce_2 = where_aoi(_df_shuffled_mce, "stockerau_full_8000_0")
    _df_unshuffled_mce_1 = where_aoi(_df_unshuffled_mce, "rdd_extent_8000_0")
    _df_unshuffled_mce_2 = where_aoi(_df_unshuffled_mce, "stockerau_full_8000_0")

    mce_plot = (
        plot_mce(_df_shuffled_mce_1, _df_unshuffled_mce_1)
        + plot_mce(_df_shuffled_mce_2, _df_unshuffled_mce_2)
    ).opts(shared_axes=False)
    return SUBPLOT_WIDTH, mce_plot, where_aoi


@app.cell
def _(
    SCL,
    SUBPLOT_WIDTH,
    item_aois,
    item_dates,
    scl_20m_dists,
    scl_colors,
    where_aoi,
):
    DIST_PLOT_HEIGHT = 170
    scl_cmap_bar = {item.name: scl_colors[item] for item in SCL}
    scl_cols = [item.name for item in SCL]


    def plot_dists(dists):
        _df = dists.join(item_dates).sort_values(by="date_str")
        return _df.hvplot.bar(y=scl_cols,
            stacked=True, cmap=scl_cmap_bar, legend=False,
        ).opts(width=SUBPLOT_WIDTH, height=DIST_PLOT_HEIGHT)

    _dists = scl_20m_dists.join(item_aois)
    _dists_1 = where_aoi(_dists, "rdd_extent_8000_0")
    _dists_2 = where_aoi(_dists, "stockerau_full_8000_0")

    dist_plot = (plot_dists(_dists_1) + plot_dists(_dists_2))#.opts(shared_axes=False)
    return dist_plot, scl_cmap_bar, scl_cols


@app.cell
def _(dist_plot, mce_plot):
    (mce_plot + dist_plot).cols(2).opts(shared_axes=False)
    return


@app.cell
def _(item_aois, item_dates, scl_20m_dists, where_aoi):
    # Output image dates
    _dists = scl_20m_dists.join(item_aois)
    _dists_1 = where_aoi(_dists, "rdd_extent_8000_0")
    _dists_2 = where_aoi(_dists, "stockerau_full_8000_0")
    _dists_2.join(item_dates)["date_str"].sort_values()
    return


@app.cell
def _(pd, shuffled_100m_mce_stats, unshuffled_100m_mce):
    mce_comp_100m = pd.DataFrame(
        {
            "unshuffled_100m_mce": unshuffled_100m_mce,
            "shuffled_100m_mce_median": shuffled_100m_mce_stats["median"],
            "diff": unshuffled_100m_mce - shuffled_100m_mce_stats["median"],
            "unshuffled_better": unshuffled_100m_mce < shuffled_100m_mce_stats["median"]
        }
    )

    mce_comp_100m
    return (mce_comp_100m,)


@app.cell
def _(mce_comp_100m):
    _df = mce_comp_100m
    _unshuffled_better = _df[_df["unshuffled_better"]]
    _n = len(_df)
    _n_unshuffled_better = len(_unshuffled_better)
    _pc_unshuffled_better = 100 * _n_unshuffled_better / _n
    _mean_mce_diff = abs(_unshuffled_better["diff"].mean())

    print(f"Of {_n} images, {_n_unshuffled_better} ({_pc_unshuffled_better:0.1f}%) had lower MCE due to downsampling than the majority of spatially shuffled permutations. Of these, the mean reduction in MCE was {_mean_mce_diff:0.2f}%.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Downsampled at 200m × 200m resolution
    """)
    return


@app.cell
def _(calc_dists_2, pd, scl_20m_dists, scl_shuffled_200m):
    shuffled_200m_dists = calc_dists_2(scl_shuffled_200m)
    shuffled_200m_mce = (
        (scl_20m_dists - shuffled_200m_dists)
        .drop(columns=["Total_pixels"])
        .abs()
        .max(axis=1)
    )

    _grp = shuffled_200m_mce.groupby("id")
    shuffled_200m_mce_stats = pd.DataFrame(
        {
            "mean": _grp.mean(),
            "std": _grp.std(),
            "min": _grp.min(),
            "max": _grp.max(),
            "median": _grp.median()
        }
    )
    shuffled_200m_mce_stats
    return shuffled_200m_dists, shuffled_200m_mce, shuffled_200m_mce_stats


@app.cell
def _(pd, shuffled_200m_mce_stats, unshuffled_200m_mce):
    mce_comp_200m = pd.DataFrame(
        {
            "unshuffled_200m_mce": unshuffled_200m_mce,
            "shuffled_200m_mce_median": shuffled_200m_mce_stats["median"],
            "diff": unshuffled_200m_mce - shuffled_200m_mce_stats["median"],
            "unshuffled_better": unshuffled_200m_mce < shuffled_200m_mce_stats["median"]
        }
    )

    mce_comp_200m
    return (mce_comp_200m,)


@app.cell
def _(mce_comp_200m):
    _df = mce_comp_200m
    _unshuffled_better = _df[_df["unshuffled_better"]]
    _n = len(_df)
    _n_unshuffled_better = len(_unshuffled_better)
    _pc_unshuffled_better = 100 * _n_unshuffled_better / _n
    _mean_mce_diff = abs(_unshuffled_better["diff"].mean())

    print(f"Of {_n} images, {_n_unshuffled_better} ({_pc_unshuffled_better:0.1f}%) had lower MCE due to downsampling than the majority of spatially shuffled permutations. Of these, the mean reduction in MCE was {_mean_mce_diff:0.2f}%.")
    return


@app.cell
def _(shuffled_200m_mce, unshuffled_200m_mce):
    (shuffled_200m_mce.hvplot.box(by="id") * unshuffled_200m_mce.hvplot.scatter(color="red")).opts(show_legend=False)
    return


@app.cell
def _(SCL, scl_colors):
    scl_cmap = {str(item.value): scl_colors[item] for item in SCL}

    def plot(da):
        return da.astype(str).hvplot.image(data_aspect=1.0, title="", cmap=scl_cmap)

    return (plot,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Single scene plots
    """)
    return


@app.cell
def _(
    mo,
    plot,
    scl_100m,
    scl_200m,
    scl_20m,
    scl_shuffled_100m,
    scl_shuffled_200m,
    scl_shuffled_20m,
):
    VIZ_ENABLED = True

    mo.stop(not VIZ_ENABLED)

    sel_id = "S2B_MSIL2A_20230605T110619_R137_T30TUM_20230605T163049"
    sel_perm = 0


    _ids = [
        #"S2B_MSIL2A_20230615T110629_R137_T30TUM_20230615T205813",
        sel_id,
    ]
    _i = sel_perm

    _inputs = [
        [
            scl_20m[id],
            scl_100m[id],
            scl_200m[id],
            scl_shuffled_20m[id][_i],
            scl_shuffled_100m[id][_i],
            scl_shuffled_200m[id][_i],
        ]
        for id in _ids
    ]

    p = plot(_inputs[0][0])
    del _inputs[0][0]

    for _row in _inputs:
        for _col in _row:
            p += plot(_col)

    # (
    #     plot(scl_20m[_i])
    #     + plot(scl_100m[_i])
    #     + plot(scl_200m[_i])
    #     + plot(scl_shuffled_20m[_i])
    #     + plot(scl_shuffled_100m[_i])
    #     + plot(scl_shuffled_200m[_i])
    # ).cols(6)

    p.cols(6)
    return sel_id, sel_perm


@app.cell
def _(scl_cmap_bar, scl_cols):
    WIDTH_SIDEPLOT = 175
    MAX_CLASS_PC = 80
    MAX_DIFF_PC = 2.0


    def single_dist_plot(df_dists):
        return df_dists.hvplot.bar(
            y=scl_cols,
            invert=True,
            xlabel="",
            ylabel="",
            width=WIDTH_SIDEPLOT,
            ylim=[0, MAX_CLASS_PC],
            title="Class proportion (%)",
            cmap=scl_cmap_bar,
        )


    def single_diff_plot(df_dists_native, df_dists_downsampled):
        _diff = abs(df_dists_native - df_dists_downsampled)
        return _diff.hvplot.bar(
            y=scl_cols,
            invert=True,
            width=WIDTH_SIDEPLOT,
            xlabel="",
            ylabel="",
            ylim=[MAX_DIFF_PC, 0],
            title="Diff. from native (%)",
            color="red",
        )


    def combo_sideplot(df_dists_native, df_dists_downsampled):
        return (
            single_diff_plot(df_dists_native, df_dists_downsampled)
            + single_dist_plot(df_dists_downsampled)
        ).opts(shared_axes=False)

    return combo_sideplot, single_dist_plot


@app.cell
def _(scl_20m_dists, sel_id, single_dist_plot):
    _df_native = scl_20m_dists.loc[[sel_id]]
    single_dist_plot(_df_native)
    return


@app.cell
def _(combo_sideplot, scl_20m_dists, sel_id, unshuffled_100m_dists):
    _df_20m = scl_20m_dists.loc[[sel_id]]
    _df_u100m = unshuffled_100m_dists.loc[[sel_id]]

    combo_sideplot(_df_20m, _df_u100m).opts(title="Unshuffled @ 100m")
    return


@app.cell
def _(combo_sideplot, scl_20m_dists, sel_id, unshuffled_200m_dists):
    _df_20m = scl_20m_dists.loc[[sel_id]]
    _df_u200m = unshuffled_200m_dists.loc[[sel_id]]

    combo_sideplot(_df_20m, _df_u200m).opts(title="Unshuffled @ 200m")
    return


@app.cell
def _(combo_sideplot, scl_20m_dists, sel_id, sel_perm, shuffled_100m_dists):
    _df_20m = scl_20m_dists.loc[[sel_id]]
    _df_s100m = shuffled_100m_dists.loc[[(sel_id, sel_perm)]]

    combo_sideplot(_df_20m, _df_s100m).opts(title=f"Shuffled (permutation {sel_perm}) @ 100m")
    return


@app.cell
def _(combo_sideplot, scl_20m_dists, sel_id, sel_perm, shuffled_200m_dists):
    _df_20m = scl_20m_dists.loc[[sel_id]]
    _df_s200m = shuffled_200m_dists.loc[[(sel_id, sel_perm)]]

    combo_sideplot(_df_20m, _df_s200m).opts(title=f"Shuffled (permutation {sel_perm}) @ 200m")
    return


if __name__ == "__main__":
    app.run()
