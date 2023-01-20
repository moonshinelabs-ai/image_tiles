python_sources()

pex_binary(
    name="image_grid_server",
    entry_point="image_grid_server.py",
    dependencies=[":templates", ":static"]
)

python_tests(
    name="tests",
)

pex_binary(
    name="glob_test",
    entry_point="glob_test.py",
)

pex_binary(
    name="file_exists_test",
    entry_point="file_exists_test.py",
)

pex_binary(
    name="image_test",
    entry_point="image_test.py",
)

resources(
    name="templates",
    sources=["templates/*"],
)

resources(
    name="static",
    sources=["static/*"],
)