import os
import sys
import subprocess
import venv


def create_venv(venv_dir):
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment at {venv_dir}")
        venv.create(venv_dir, with_pip=True)
    else:
        print(f"Virtual environment already exists at {venv_dir}")


def install_requirements(requirements_file, venv_dir):
    """Install requirements from the requirements.txt file using the virtual environment's pip."""
    if os.path.exists(requirements_file):
        # Use the virtual environment's pip to install the requirements
        if os.name == 'nt':
            pip_path = os.path.join(venv_dir, 'Scripts', 'pip')
        else:
            pip_path = os.path.join(venv_dir, 'bin', 'pip')

        subprocess.check_call([pip_path, 'install', '-r', requirements_file])
    else:
        raise FileNotFoundError(f"Requirements file not found at {requirements_file}")


def main():
    venv_dir = '.venv'
    requirements_file = 'requirements.txt'

    create_venv(venv_dir)
    install_requirements(requirements_file, venv_dir)
    print("Setup complete. Virtual environment is ready and requirements are installed.")


if __name__ == '__main__':
    main()
