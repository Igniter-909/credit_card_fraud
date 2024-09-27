from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="creditCardFraudDetection",
    version="0.0.1",
    author="Roshan Kumar Sahu",
    author_email = "igniterofficial909505@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)