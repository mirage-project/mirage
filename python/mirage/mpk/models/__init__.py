import importlib
from pathlib import Path

package_dir = Path(__file__).parent

for path in package_dir.rglob("builder.py"):

    relative_path = path.relative_to(package_dir).with_suffix("")

    module_name = str(relative_path).replace('\\', '/').replace('/', '.')

    try:
        importlib.import_module(f".{module_name}", package=__name__)
    except ImportError as e:
        # error out
        print(f"Failed to import module {module_name}: {e}")
        raise e