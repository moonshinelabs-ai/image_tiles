import argparse
import io
import os
import random
from typing import Optional

import flask
import imageio.v3 as imageio
from loguru import logger

from .file_exists import exists
from .glob import glob
from .image import get_supported_extensions, read_image


def parse_image_tile_args():
    parser = argparse.ArgumentParser(
        description="Simple server visualizes images in a folder."
    )
    parser.add_argument("folder_path", help="Which folder to serve.")
    parser.add_argument(
        "--bind",
        help="Which address to bind to, 0.0.0.0 will bind to all IPs.",
        default="0.0.0.0",
    )
    parser.add_argument("--port", help="Port to bind on.", type=int, default=8000)
    parser.add_argument(
        "--render_method",
        default="rgb",
        nargs="?",
        choices=["rgb", "bgr", "bw", "sentinel"],
        help="Rendering method, see README for details.",
    )
    parser.add_argument(
        "--normalization",
        default="standard",
        nargs="?",
        choices=["standard", "scaling", "sigmoid", "sentinel"],
        help="Normalization method, see README for details.",
    )
    parser.add_argument(
        "--debug", help="Enable server Flask debugging.", type=bool, default=False
    )
    parser.add_argument(
        "--root",
        help="Root directory to serve templates/static from.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--num_items", help="Maximum images to show on the page.", type=int, default=100
    )

    args = parser.parse_args()

    # Do some surgery for relative paths
    if not args.folder_path.startswith("/"):
        if "s3://" not in args.folder_path and "gs://" not in args.folder_path:
            args.folder_path = os.path.join(os.getcwd(), args.folder_path)

    return args


def get_image_files(folder: str) -> list[str]:
    """Get a list of supported image files in a folder."""
    supported_extensions = get_supported_extensions()

    # We grab the filelist once and then test if a file is valid. This
    # costs more memory but the cost of a large glob in the cloud is high.
    path = os.path.join(folder, "*")
    logger.info("Globbing files, this may take a while for a large AWS/GCP bucket...")
    filelist = glob(path)

    imagelist = []
    for f in filelist:
        ext = "." + f.split(".")[-1]
        if ext in supported_extensions:
            imagelist.append(f)

    logger.info(f"Got {len(imagelist)} items from {folder}.")
    return imagelist


def read_and_render_image(path: str, args: argparse.Namespace) -> Optional[bytes]:
    """Reads an image and converts it to a png for display in the web
    browser."""
    # Rewrite based on the input
    if path.startswith("/s3"):
        path = path.replace("/s3/", "s3://")
    elif path.startswith("/gs"):
        path = path.replace("/gs/", "gs://")

    if not exists(path):
        return None

    array = read_image(
        path, normalize=args.normalization, render_method=args.render_method
    )

    with io.BytesIO() as f:
        imageio.imwrite(uri=f, image=array, extension=".png")  # type: ignore
        return f.getvalue()


def server_path(ext_path: str) -> str:
    """Convert an external path to one we can use internally."""
    if "s3://" in ext_path:
        return f"images/s3/{ext_path[5:]}"
    elif "gs://" in ext_path:
        return f"images/gs/{ext_path[5:]}"
    else:
        return f"images/{ext_path[1:]}"


def flask_app(args):
    """Defines the flask app itself."""
    root = os.path.join(os.getcwd(), args.root) if args.root else ""
    app = flask.Flask(
        __name__,
        template_folder=os.path.join(root, "templates"),
        static_folder=os.path.join(root, "static"),
    )

    @app.route("/images/<path:image_path>")
    def image(image_path):
        full_path = f"/{image_path}"
        data = read_and_render_image(full_path, args)

        return flask.Response(data, mimetype="image/png")

    @app.route("/")
    def index():
        url_args = flask.request.args

        num_items = args.num_items
        if "num_items" in url_args:
            num_items = int(url_args["num_items"])

        logger.info(f"Reading from folder {args.folder_path}")
        filelist = get_image_files(args.folder_path)
        random.shuffle(filelist)
        filelist = filelist[0:num_items]
        items = [server_path(f) for f in filelist]
        logger.info(f"Request recieved, found {len(filelist)} items to display")

        return flask.render_template(
            "image_tiles.html",
            folder_location=args.folder_path,
            num_images=num_items,
            items=items,
            normalization=args.normalization,
            render_method=args.render_method,
        )

    return app


def image_tile_server():
    args = parse_image_tile_args()
    logger.info("Starting image_tiles server on http://%s:%s" % (args.bind, args.port))
    app = flask_app(args)
    app.run(host=args.bind, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    image_tile_server()
