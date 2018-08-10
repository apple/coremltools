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
        Path('ci/.pypirc').rename(Path.home() / '.pypirc')
        setup('upload', input=pypi_password.encode('utf-8'))
    else:
        setup()

if __name__ == '__main__':
    main()
