from setuptools import setup

setup(name='placetextblockonimage',
version='0.0.1',
install_required=['Pillow','pydantic', 'torch==1.12.1', 'gym==0.21.0', 'stable_baselines3'],
description='This is a package for training a model to place text on an image without overlapping with other text',
readme='README.md',
)