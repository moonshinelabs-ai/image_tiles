# Image Tiles

*A Moonshine Labs tool*

## Overview
A simple but flexible tool to view a folder full of images on your web browser. Features:

* Run one command and serve a folder of images in any format PIL supports.
* Easily view images on another computer, such as when working via SSH or remotely.
* Support for AWS (and eventually GCS/Azure) buckets.
* Normalization and rendering options for a variety of multispectral images, especially satellites.
* Support for multichannel TIFF, JP2, and other less common file formats.

## Installation
```
# Install default installation
$ pip install image_tiles

# Install with S3 support
$ pip install image_tiles[aws]
```

## Usage
```
$ image_tiles ./path_to_folder
```

### Serving a folder of JPEGs (images from instagram.com/dustinlefevre)

![example_page](docs_images/image_tiles_demo.png)

### Serving a folder of multispectral TIFFs:

![example_sat](docs_images/image_tiles_sat.png)

### Rendering options
* `rgb`: Standard RGB image rendering (default).
* `bgr`: BGR image rendering.
* `bw`: Grayscale image rendering from the first 3 channels.
* `sentinel`: Render using channels[1:4] for sentinel satellite data.

### Normalization options
* `standard`: If the image is a standard 1/3 channel image, leave it alone. Otherwise apply `scaling` (default)
* `scaling`: Scale to 0-255, clipping negative numbers and scaling positive numbers.
* `sigmoid`: Sigmoid normalization, as described in [xarray true color](https://xarray-spatial.org/reference/_autosummary/xrspatial.multispectral.true_color.html)
* `sentinel`: Sentinel-2 specific normalization, as described at the Sentinel 2 [user guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/definitions)

## Current Limitations

* Only a subset of useful normalization and rendering options supported. Help contribute!
* Eventually we'd like to more easily support user code and functions.
* Must restart the server to change some options.