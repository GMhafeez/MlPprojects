from setuptools import find_packages, setup

from typing import List

Hypen_E_Dor = "-e ."

def get_requirements(filename:str)-> List[str]:
     "This function is used to read the requirement.txt file"
     requirement = []
     with open('requirement.txt') as file_obj :
          requirement = file_obj.readlines()
          [req.replace("\n","")for req in requirement]

          if Hypen_E_Dor in requirement:
               requirement.remove(Hypen_E_Dor)

               return requirement

setup(
    name= 'ML project',
    version= '0.0.1',
    author= 'Ghulam Mustafa',
    author_email= 'gmhafeez17@gmail.com',
    packages= find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
    