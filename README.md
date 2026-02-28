# eaggq-materials

Source code and intermediate outputs for the paper _Efficient aggregate land cover queries with cloud-optimized raster formats (extended version)_, by Luke McQuade, Martin Sudmanns and Dirk Tiede, of the [EO Analytics group](https://www.plus.ac.at/geoinformatik/research/research-areas/eo-analytics/?lang=en), Department of Geoinformatics—Z_GIS, University of Salzburg.


## Overview

This repository consists of several [marimo](https://marimo.io) notebooks:
- `sentinel2_search.py`: A parameterized notebook that performs a simple content-based search of Sentinel-2 imagery using the [Scene Classification Layer](https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-L2AAlgorithmsS2-Processing-L2A-Algorithmstrue), with the [Microsoft Planetary Computer API]().
- `downsampling_analysis.py`: The main notebook, which runs simulated queries against many areas of interest (AoIs) at various spatial resolutions, and analyses the result quality and runtime performance.
- `spatial_shuffle_analysis.py`: A supplementary notebook demonstrating the effect of spatial distribution on downsampling quality performance.

Other key files:  
- `utils.py`: A Python module containing utility functions used by the notebooks.
- `pixi.toml` specifies the project dependencies.
- `docker-compose.yml` and `Dockerfile` provide a containerized environment.
- `data/dsa_results/*.csv`: Intermediate result files.
- `aois.gpkg`: The collection of generated AoIs queried against. Optionally exported 
during analysis.


## Requirements

Hardware: a PC with at least 16GB memory.

Software: Linux (host or virtual machine); a Docker-compatible container host.

Note: Under the hood, this project uses [pixi](https://pixi.prefix.dev/latest/) as a package/environment manager; it may be possible to use this directly in your host OS instead of the Docker container, but this has not been tested.


## Getting started

⚠️ This opens a notebook server without authentication, make sure to be behind a properly configured firewall.

Get started with, e.g., `docker compose up --build`, and browse to:
* `http://localhost:8080/?file=downsampling_analysis.py`, for the main downsampling analysis notebook, or,
* `http://localhost:8080/?file=spatial_shuffle_analysis.py`, for the spatial distribution demonstration notebook.




## Acknowledgements

Many thanks to our colleague, Felix Kröber (@fkroeber), whose earlier explorations and Jupyter notebooks on Sentinel-2 STACs and the SCL were a great help. Many thanks also to the Microsoft Planetary Computer team, and the authors and maintainers of the many open source packages used in this project.


## Funding

The research leading to these results has received funding from the European Union's Horizon Europe research and innovation program under the Grant Agreement No. 101082493 (Project: LEONSEGS).
