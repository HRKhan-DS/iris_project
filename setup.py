from setuptools import setup, find_packages


with open('requirements.txt', 'r') as file:
      requirement = file.read().splitlines()


setup(
      name= 'Simple Iris Classification',
      version= '0.0.1',
      author= 'Md.Harun-Or-Rashid Khan', 
      author_email= 'mdhrkhandata.analyst@gmail.com',
      install_requires= requirement,
      packages= find_packages()
      )
