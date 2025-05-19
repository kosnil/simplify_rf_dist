from setuptools import setup, find_packages

setup(
    name='TopKRandomForest',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'joblib==1.4.0', 'matplotlib==3.8.4', 'numba==0.59.1', 'numpy==1.26.4', 'pandas==2.2.3', 'scikit_learn==1.4.2',
        'scipy==1.15.3', 'seaborn==0.13.2', 'tqdm==4.66.2'
    ],
    author='Nils Koster, Fabian Krüger',
    author_email='nils.koster@kit.edu',
    description='Topk Random Forest for simplifying RF forecast distributions and improving interpretability',
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kosnil/simplify_rf_dist',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.10',
)
