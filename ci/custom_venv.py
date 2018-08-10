import venv
import subprocess


class CustomVenv(venv.EnvBuilder):
    '''Virtual environment for the installation.

    Install wheel in the venv for packaging.
    '''
    def __init__(self, *args, **kwargs):
        self.requirements = kwargs.pop('requirements')
        super().__init__(*args, **kwargs)

    def post_setup(self, context):
        if not self.requirements:
            return
        self.python = context.env_exe
        def pip(*args):
            subprocess.check_call([self.python, '-m', 'pip'] + list(args))
        pip('install', '-U', 'pip')
        pip('install', *self.requirements)
