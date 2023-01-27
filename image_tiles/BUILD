python_sources()

pex_binary(
    name="image_tile_server",
    entry_point="image_tile_server.py",
    dependencies=[":templates", ":static"]
)

python_tests(
    name="tests",
)

resources(
    name="templates",
    sources=["templates/*"],
)

resources(
    name="static",
    sources=["static/*"],
)