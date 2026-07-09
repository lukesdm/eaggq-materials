import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

with app.setup:
    # For the paper 'Efficient semantic content-based image retrieval with cloud-optimized raster formats'
    # Authors: Luke McQuade, Martin Sudmanns, Dirk Tiede
    # February 2026

    import marimo as mo

    import logging

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rioxarray
    import shapely
    import shapely.geometry as sg
    import stackstac

    from utils import ContentBasedSearchParams, SCL, scl_colors, find_stac_items

    logger = logging.getLogger(__name__)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Semantic content-based search for Sentinel-2

    This notebook performs a content-based search using the Scene Classification Layer (SCL), part of the Level 2A product. See the [Microsoft Planetary Computer dataset](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) and [Sentinel-2 Level-2A Algorithm Theoretical Basis Document](https://step.esa.int/thirdparties/sen2cor/2.10.0/docs/S2-PDGS-MPC-L2A-ATBD-V2.10.0.pdf). It is parameterized, allowing it to be used from a parent notebook or module.

    if running from a parent notebook, pass in the `args` variable, which should be an instance of `utils.ContentBasedSearchParams`.
    the distributions of the SCL classes can be accessed by the `class_distribution` variable, and matching results (i.e., filtered according to the input criteria) with the `filtered_items` variable.

    Filtering supported:
    - Cloud/shadow/snow (combined), with the `args.max_cloud` parameter.
    - Vegetation, with the `args.min_veg` and `args.max_veg` parameters.
    - Bare soil, with the `args.min_soil` and `args.max_soil` parameters.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare inputs
    """)
    return


@app.cell
def _(args):
    def has_args():
        """Test for whether args have been passed into a notebook by a parent."""
        # Note: Need to keep this in each notebook that uses it.
        if "args" in globals():
            assert isinstance(args, ContentBasedSearchParams)
            return True
        else:
            return False

    return (has_args,)


@app.cell
def _(has_args):
    if not has_args():
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s"
        )
        logging.getLogger(__name__).setLevel(logging.INFO)
    return


@app.cell
def _(args, has_args):
    # Stockerau, Austria
    _aoi_test = shapely.from_wkt(
        "POLYGON ((16.17 48.33, 16.25 48.33, 16.25 48.41, 16.17 48.41, 16.17 48.33))"
    )

    if has_args():
        area_of_interest = args.aoi
    else:
        logger.info("No AoI specified, using test area.")
        area_of_interest = _aoi_test

    aoi_gdf = gpd.GeoDataFrame(geometry=[sg.shape(area_of_interest)], crs=4326)
    return aoi_gdf, area_of_interest


@app.cell
def _(area_of_interest, args, has_args):
    if has_args():
        items = args.stac_items
    else:
        logger.info("No stac items specified, using test search parameters.")
        _test_period = "2023-06-01/2023-07-01"
        items = find_stac_items(area_of_interest, _test_period, method="contains")
        logger.info(f"Found {len(items)} items.")
    return (items,)


@app.cell
def _(items):
    if items[0].properties.get("proj:code") is not None:
        epsg = int(items[0].properties["proj:code"][5:])
    elif items[0].properties.get("proj:epsg") is not None:
        epsg = items[0].properties["proj:epsg"]
    else:
        raise Exception("Couldn't determine EPSG code from STAC items.")

    logger.info(f"Using EPSG code (of first image found): {epsg}")
    return (epsg,)


@app.cell
def _(aoi_gdf, epsg):
    _minx, _miny, _maxx, _maxy = aoi_gdf.to_crs(epsg).total_bounds
    width = _maxx - _minx
    height = _maxy - _miny

    logger.info(f"AOI width={width:.2f}, height={height:.2f}")
    return


@app.cell
def _(args, has_args):
    if has_args() and args.cbs_resolution is not None:
        resolution = args.cbs_resolution
    else:
        resolution = 20
    return (resolution,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(aoi_gdf, epsg, items, resolution):
    stack = stackstac.stack(
        items,
        assets=["SCL"],
        epsg=epsg,
        resolution=resolution,
        bounds_latlon=aoi_gdf.total_bounds,
    )

    # Clip to the actual AoI geometry, rather than just the bounding rectangle.
    stack = stack.rio.clip(aoi_gdf.to_crs(epsg).geometry)
    return (stack,)


@app.cell
def _(stack):
    stack
    return


@app.cell
def _(stack):
    data = stack.compute(scheduler="synchronous")

    # Clipping introduces NaN's, which we want to align with SCL's nodata value,
    # otherwise category percentages aren't calculated correctly.
    data = data.fillna(0)
    return (data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calculate class distribution
    """)
    return


@app.function
def count_cats(arr, n_cats=12):
    cat_bins = np.arange(0, n_cats + 1)
    counts = np.histogram(arr.values, bins=cat_bins)[0]
    total = max(
        arr.size - counts[0], 1
    )  # Total excluding NoData; set to 1 in empty case.
    df = pd.DataFrame(
        {
            "category": cat_bins[:-1],
            "count": counts,
            "pc": 100.0 * counts / total,
        }
    )
    df.loc[0, "pc"] = np.nan  # Exclude NoData
    return df


@app.cell
def _(data):
    _cat_counts = []

    for _scene in data:
        _df = count_cats(_scene)
        _id = str(_scene["id"].values)
        _df["id"] = _id
        _df = _df.pivot(index="id", columns="category", values="pc")
        _df.columns = [item.name for item in SCL]
        _df["Total_pixels"] = _scene.size
        _df["NoData_pixels"] = (_scene == 0).sum().values
        _df["NoData_pc"] = 100 * _df["NoData_pixels"] / _scene.size
        _cat_counts.append(_df)

    class_distribution = pd.concat(_cat_counts)
    return (class_distribution,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Perform filtering
    """)
    return


@app.cell
def _(args, class_distribution, has_args, items):
    # ⚠ Output will be read from here, be careful changing things around here.

    if has_args():
        _max_cloud = args.max_cloud
        _min_veg = args.min_veg
        _max_veg = args.max_veg
        _min_soil = args.min_soil
        _max_soil = args.max_soil
    else:
        print("No filter parameters specified, using test values.")
        _max_cloud = 100.0
        _min_veg = 0.0
        _max_veg = 100.0
        _min_soil = 0.0
        _max_soil = 100.0

    # Cloud and snow are not well separated in the SCL
    _cloud_filt = (
        class_distribution[SCL.Cl_Cirrus.name]
        + class_distribution[SCL.Cl_Med.name]
        + class_distribution[SCL.Cl_High.name]
        + class_distribution[SCL.Snow.name]
        + class_distribution[SCL.Shadow.name]
    ) < _max_cloud

    _veg_filt = (class_distribution[SCL.Veg.name] > _min_veg) & (
        class_distribution[SCL.Veg.name] < _max_veg
    )

    _soil_filt = (class_distribution[SCL.Bare.name] > _min_soil) & (
        class_distribution[SCL.Bare.name] < _max_soil
    )

    _filtered = class_distribution[_cloud_filt & _veg_filt & _soil_filt]

    # Return matching STAC items
    filtered_items = [item for item in items if item.id in _filtered.index]
    return


@app.cell
def _():
    mo.md(r"""
    ## Visualization/EDA
    """)
    return


@app.cell
def _(has_args):
    viz_enabled = (
        mo.running_in_notebook() and
        # Assume running in headless mode if args have been supplied, even from another notebook.
        not has_args()
    )
    return (viz_enabled,)


@app.cell
def _(class_distribution, viz_enabled):
    mo.stop(not viz_enabled)

    class_distribution
    return


@app.cell
def _(viz_enabled):
    mo.stop(not viz_enabled)

    import holoviews as hv
    import hvplot.xarray
    import hvplot.pandas

    hv.extension("bokeh")
    return (hv,)


@app.cell
def _(class_distribution, viz_enabled):
    mo.stop(not viz_enabled)

    _scl_cmap = {item.name: scl_colors[item] for item in SCL}
    _scl_labels = {item.value: item.name for item in SCL if item != SCL.Mask}
    _ds = class_distribution[list(_scl_labels.values())].copy()
    _ds["date_str"] = _ds.index.str.slice(11, 19)
    _ds = _ds.sort_values(by="date_str")

    _ds.hvplot.bar(x="date_str", stacked=True).opts(
        xlabel="Image date",
        xrotation=90,
        ylabel="%",
        height=400,
        title="Sentinel-2 image class distribution",
        cmap=_scl_cmap,
    )
    return


@app.cell
def _(aoi_gdf, epsg, items, resolution, viz_enabled):
    mo.stop(not viz_enabled)

    rgb_stack = stackstac.stack(
        items,
        assets=["B04", "B03", "B02"],
        epsg=epsg,
        resolution=resolution,
        bounds_latlon=aoi_gdf.total_bounds,
        properties=False,
    )

    # Clip to the actual AoI geometry, rather than just the bounding rectangle.
    rgb_stack = rgb_stack.rio.clip(aoi_gdf.to_crs(epsg).geometry)

    rgb_data = rgb_stack.compute()
    return (rgb_data,)


@app.cell
def _(hv, viz_enabled):
    mo.stop(not viz_enabled)

    # Via ChatGPT o3 (16-Jul-2025)...
    def scl_vertical_legend(colors, box_size=1.0, pad=0.3):
        rects, labels = [], []
        for idx, scl in enumerate(sorted(SCL, key=lambda s: s.value)):
            top, bot = -idx * box_size, -(idx + 1) * box_size
            rects.append(
                hv.Rectangles([(0, bot, box_size, top)])
                  .opts(color=colors[scl], line_color='black', line_width=0.5)
            )
            labels.append(
                hv.Text(x=box_size + pad, y=(top + bot) / 2, text=scl.name)
                  .opts(text_align='left', text_baseline='middle',
                        text_color='black', text_font_size='10pt')
            )

        legend = (hv.Overlay(rects) * hv.Overlay(labels)).opts(
            xlim=(-0.1, box_size + pad + 3),      # <— widen x-range so labels fit
            ylim=(-(len(SCL))*box_size, 0),       # tidy y-range
            xaxis=None, yaxis=None, show_frame=False,
            width=240, height=int(box_size*len(SCL)*30),
            hooks=[lambda p, _: setattr(p.state.toolbar, 'logo', None)]
        )
        return legend

    return (scl_vertical_legend,)


@app.cell
def _(items, viz_enabled):
    mo.stop(not viz_enabled)

    _options = [item.id for item in items]
    ui_img_sel = mo.ui.dropdown(options=_options, value=_options[0])
    ui_img_sel
    return (ui_img_sel,)


@app.cell
def _(data, rgb_data, scl_vertical_legend, ui_img_sel, viz_enabled):
    mo.stop(not viz_enabled)

    _sel_rgb_data = rgb_data[rgb_data.id == ui_img_sel.value][0]
    _sel_rgb_data = _sel_rgb_data**0.6  # Quick and dirty scaling for viz
    _rgb_plot = _sel_rgb_data.hvplot.rgb(x="x", y="y", bands="band", data_aspect=1)

    _scl_cmap = {str(item.value): scl_colors[item] for item in SCL}
    _sel_scl_data = data[data.id == ui_img_sel.value][0][0]
    _sel_scl_data = _sel_scl_data.astype(np.uint8).astype(str)
    _scl_plot = _sel_scl_data.hvplot.image(
        x="x", y="y", cmap=_scl_cmap, data_aspect=1
    )
    _scl_legend = scl_vertical_legend(scl_colors, box_size=0.8).opts(width=130)

    (_rgb_plot + _scl_plot + _scl_legend)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
