import os
import sys
import subprocess
import venv

def create_venv(venv_dir):
  if not os.path.exists(venv_dir):
    venv.create(venv_dir, with_pip=True)
    

def activate_venv(venv_dir):
  if os.name == 'nt':
    activate_script = os.path.join(venv_dir, 'Scripts', 'activate')
  else:
    activate_script = os.path.join(venv_dir, 'bin', 'activate')
  
  if not os.path.exists(activate_script):
    raise FileNotFoundError(f'Virtual environment script not found at: {activate_script}')

  exec(open(activate_script).read(), {'__file__': activate_script})


def install_requirements(requirements_file):
  if os.path.exists(requirements_file):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
  else:    
    raise FileNotFoundError(f'Requirements file not found at: {requirements_file}')


def main():
  venv_dir = '.venv'
  requirements_file = 'requirements.txt'
  create_venv(venv_dir)
  activate_venv(venv_dir)
  install_requirements(requirements_file)
  print('Environment setup complete.')


if __name__ == '__main__':
  main()
