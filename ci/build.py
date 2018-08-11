import os
import subprocess
import sys

from custom_venv import CustomVenv


def main():
    env = CustomVenv(clear=True, symlinks=True, with_pip=True, requirements=['wheel'])
    env.create('venv')

    def setup(*args, **kwargs):
        subprocess.run(
            [env.python, '../patched_setup.py', 'bdist_wheel', '--plat-name', 'win_amd64', '--python-tag', 'py{}{}'.format(*sys.version_info[:2]), *args],
            cwd='coremltools', check=True, **kwargs)

    pypi_password = os.environ.get('pypi_password')
    if pypi_password:
        from pathlib import Path
        with open('ci/.pypirc') as base_pypirc, (Path.home() / '.pypirc').open('w') as pypirc:
            pypirc.write(base_pypirc.read())
            pypirc.write('password: {}\n'.format(pypi_password))
        setup('upload')
    else:
        setup()

if __name__ == '__main__':
    main()
