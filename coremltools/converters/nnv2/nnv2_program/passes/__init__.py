#from .const_elimination import const_elimination
#from .matmul_to_linear import matmul_to_linear
#from .dead_code_elimination import dead_code_elimination
#from . import *
from os.path import dirname, basename, isfile, join
import glob

# Import all passes
excluded_files = [
        'pass_registry.py',
        'common_pass.py',
        '__init__.py']
modules = glob.glob(join(dirname(__file__), "*.py"))
pass_modules = [basename(f)[:-3] for f in modules if \
        isfile(f) and \
        # Follow python convention to hide _* files.
        basename(f)[:1] != '_' and \
        basename(f)[:4] != 'test' and \
        basename(f) not in excluded_files]
__all__ = pass_modules
from . import *  # import everything in __all__
