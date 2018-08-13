import distutils.util
import os
import subprocess
import sys

from custom_venv import CustomVenv


def get_platform():
    return distutils.util.get_platform().replace('-', '_').replace('.', '_')


def copy_credentials(password):
    '''Set the PyPI configuration and get the password from a secured environment variable.'''
    from pathlib import Path
    with open('ci/.pypirc') as base_pypirc, (Path.home() / '.pypirc').open('w') as pypirc:
        pypirc.write(base_pypirc.read())
        pypirc.write('password: {}\n'.format(password))


def main():
    env = CustomVenv(clear=True, symlinks=True, with_pip=True, requirements=['wheel'])
    env.create('venv')

    def setup(*args):
        subprocess.run([
            env.python, 'setup.py', 'bdist_wheel',
            '--plat-name', get_platform(),
            '--python-tag', 'cp{}{}'.format(*sys.version_info[:2]),
            *args,
        ], check=True)

    pypi_password = os.environ.get('pypi_password')
    if pypi_password:
        copy_credentials(pypi_password)
        setup('upload')
    else:
        # Build but don't upload if no password is provided
        setup()

if __name__ == '__main__':
    main()
