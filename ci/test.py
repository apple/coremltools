import os
import subprocess
import sys
import traceback

from custom_venv import CustomVenv

def run_tests():
    env = CustomVenv(clear=True, symlinks=True, with_pip=True, requirements=['-r', 'ci/test_requirements.txt'])
    env.create('venv')
    def python(*args):
        subprocess.check_call([env.python, '-m'] + list(args))
    python('pytest', 'coremltools')


def main():
    if sys.version_info >= (3, 7) or sys.version_info < (3, 5) and os.name == 'nt':
        print('Some dependencies are unavailable for this Python version in this system, tests are expected to fail')
        try:
            run_tests()
        except Exception:  # Don't fail the CI build
            traceback.print_exc()
    else:
        run_tests()

if __name__ == '__main__':
    main()
