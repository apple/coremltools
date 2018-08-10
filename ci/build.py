import subprocess

from custom_venv import CustomVenv


def main():
    env = CustomVenv(clear=True, symlinks=True, with_pip=True, requirements=['wheel'])
    env.create('venv')
    subprocess.check_call([env.python, '../patched_setup.py', 'bdist_wheel'], cwd='coremltools')

if __name__ == '__main__':
    main()
