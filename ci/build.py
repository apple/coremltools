import os
import subprocess

from custom_venv import CustomVenv


def main():
    env = CustomVenv(clear=True, symlinks=True, with_pip=True, requirements=['wheel'])
    env.create('venv')

    def setup(*args, **kwargs):
        subprocess.run([env.python, '../patched_setup.py', 'bdist_wheel', *args], cwd='coremltools', check=True, **kwargs)

    pypi_password = os.environ.get('pypi_password')
    if pypi_password:
        from pathlib import Path
        with open('ci/.pypirc') as base_pypirc, (Path.home() / '.pypirc').open('w') as pypirc:
            pypirc.write(base_pypirc.read())
            pypirc.write('password: {}\n'.format(pypi_password))
        # For some reason coremltools hardcodes the wheel tag to py2.7 in setup.cfg, even in Python 3 installs.
        # We have to get rid of it, else PyPI and pip will believe the target is Python 2.7 regardless of where it was built.
        Path('coremltools/setup.cfg').unlink()
        setup('upload')
    else:
        setup()

if __name__ == '__main__':
    main()
