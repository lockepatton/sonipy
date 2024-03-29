import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
     name='sonipy',
     version='1.1',
     author='Locke Patton',
     author_email='locke.patton@cfa.harvard.edu',
     description='Sonification of 2D plots',
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://github.com/lockepatton/sonipy',
     packages=setuptools.find_packages(),
     classifiers=[
         'Programming Language :: Python :: 3.6',
         'Programming Language :: Python :: 3.7',
         'Programming Language :: Python :: 3.8',
         'Programming Language :: Python :: 3.9',
         'License :: OSI Approved :: MIT License',
         'Operating System :: OS Independent',
     ],
     install_requires=['numpy>=1.16.5', 'matplotlib', 'seaborn', 'pandas', 'scipy', 'pydub','IPython'])
