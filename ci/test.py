import os
import subprocess
import sys
import traceback

from custom_venv import CustomVenv


def run_tests():
    env = CustomVenv(clear=True, symlinks=True, with_pip=True, requirements=['-r', 'test_requirements.pip'])
    env.create('venv')

    cmake_environment = os.environ.copy()
    cmake_environment['CMAKE_BUILD_TYPE'] = 'Release'
    cmake_environment['PATH'] = os.pathsep.join((os.path.dirname(env.python), cmake_environment['PATH']))
    cmake_environment['PYTHONPATH'] = os.pathsep.join((os.getcwd(), cmake_environment.get('PYTHONPATH', '')))
    subprocess.run(['cmake', '.'], check=True, env=cmake_environment)
    subprocess.run(['cmake', '--build', '.', '--target', 'pytest', '--config', 'Release'], check=True, env=cmake_environment)


def tests_should_pass():
    if sys.version_info >= (3, 7) or sys.version_info < (3, 5) and os.name == 'nt':
        print('Some dependencies are unavailable for this Python version in this system, tests are expected to fail')
        return False
    if os.name == 'nt':
        print('Native compilation is failing in Windows, tests are expected to fail')
        return False
    return True


def main():
    if tests_should_pass():
        run_tests()
    else:
        try:
            run_tests()
        except Exception:  # Don't fail the CI build
            traceback.print_exc()

if __name__ == '__main__':
    main()
