from setuptools import setup, find_packages

setup(
    name='TopKRandomForest',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    author='Nils Koster, Fabian Krüger',
    author_email='nils.koster@kit.edu',
    description='A short description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kosnil/simplify_rf_dist',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.10',
)
