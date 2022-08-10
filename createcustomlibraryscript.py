import os
import click

@click.command()
def createPackage():
    pacakge_name= click.prompt('Please enter the name of the package', default='delta')
    version = click.prompt('Please enter the version of the package', default='0.0.1')
    description = click.prompt('Please enter the description of the package', default='delta is a package to create a package')
    author = click.prompt('Please enter the author of the package', default='delta')
    env_name = click.prompt('Please enter the name of the environment', default='deltaenv-v0')
    env_file_name = click.prompt('Please enter the name of the environment file', default='deltaenv')
    env_file_class=click.prompt('Please enter the class of the environment file', default='DeltaEnv')
    other_package_name = click.prompt('Please enter the name of the other package', default='utils')
    os.mkdir(f'{pacakge_name}_package')
    os.chdir(f'{pacakge_name}_package')
    os.system('touch README.md')
    os.system('touch requirements.txt')
    os.system('touch setup.py')
    os.system('touch LICENSE.md')
    with open('setup.py', 'w') as f:
        f.write('from setuptools import setup, find_packages\n')
        f.write('''setup(name="{}", 
        version="{}", 
        description="{}",
        author="{}",
        install_required="['Pillow', 'numpy']",
        readme="README.md",
        packages=find_packages())'''.format(pacakge_name, version, description, author))

    os.mkdir(pacakge_name)
    with open(f'{pacakge_name}/__init__.py', 'w') as f:
        f.write('from gym.envs.registration import register\n')
        f.write('register(id="{}", entry_point="{}.envs:{}", max_episode_steps = 100)'.format(env_name, pacakge_name, env_file_class))
    os.chdir(pacakge_name)
    os.mkdir('envs')
    os.mkdir(other_package_name)
    os.system(f'touch {other_package_name}/__init__.py')
    with open(f'{other_package_name}/__init__.py', 'w') as f:
        f.write(f'from {pacakge_name}.{other_package_name}.main import *')
    os.system(f'touch {other_package_name}/main.py')

    with open(f'{other_package_name}/main.py', 'w') as f:
        f.write('def hello():\n')
        f.write('    print("hello")')
    with open('envs/__init__.py', 'w') as f:
        f.write(f'from {pacakge_name}.envs.{env_file_name} import {env_file_class}')

    with open(f'envs/{env_file_name}.py','w') as f:
        f.write('import gym\n')
        f.write('class {}(gym.Env):\n'.format(env_file_class))
        f.write('    def __init__(self):\n')
        f.write('        pass\n')
        f.write('    def step(self, action):\n')
        f.write('        pass\n')
        f.write('    def reset(self):\n')
        f.write('        pass\n')
        f.write('    def render(self):\n')
        f.write('        pass\n')
        f.write('    def close(self):\n')
        f.write('        pass\n')
        
if __name__  == '__main__':
    # os.system('rm -rf delta_package')
    createPackage()
    # usage:
    # cd to the package directory
    # pip install -e .
    # in python:
    #from package import subpackage as subpackage
    #subpackage.hello()
    
    # if custom reinforment learning environment also create:
    # import package
    # import gym
    # env = gym.make(env_name)
    # print(env.action_space)


