import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
     name='sonipy',
     version='0.1',
     author='Locke Patton',
     author_email='locke.patton@cfa.harvard.edu',
     description='Sonification of 2D plots',
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://github.com/lockepatton/sonipy',
     packages=['sonipy'],
     classifiers=[
         'Programming Language :: Python :: 2',
         'License :: OSI Approved :: MIT License',
         'Operating System :: OS Independent',
     ],
     install_requires=['numpy', 'matplotlib', 'pandas', 'pydub',
                       'repo @ https://github.com/AllenDowney/ThinkDSP.zip#egg=repo-1.0.0'],
 )
