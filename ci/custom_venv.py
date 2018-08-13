import venv
import subprocess


class CustomVenv(venv.EnvBuilder):
    '''Virtual environment for the installation.

    This class receives an additional "requirements" parameter, where PyPI packages can be
    specified to be installed in the venv.
    '''
    def __init__(self, *args, **kwargs):
        self.requirements = kwargs.pop('requirements')
        super().__init__(*args, **kwargs)

    def post_setup(self, context):
        self.python = context.env_exe
        if not self.requirements:
            return
        def pip(*args):
            subprocess.run([self.python, '-m', 'pip', *args], check=True)
        pip('install', '-U', 'pip')
        pip('install', *self.requirements)
