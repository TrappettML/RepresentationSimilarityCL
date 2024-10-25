from setuptools import setup, find_packages

setup(
    name='CLRepSim',  # Replace with your own package name
    version='0.1.0',  # Initial development version
    author='Matthew Trappett',
    author_email='mtrappet@uoregon.edu',
    description='Experiments for comparing Representations in the CL settings',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This is important if your README is Markdown
    # url='http://github.com/yourusername/yourpackagename',  # Project home page or repository URL
    packages=find_packages(exclude=('tests', 'docs', 'examples')),  # Automatically find your packages
    install_requires=[
        'numpy',  # Example library, replace with your dependencies
        'matplotlib',  # Another example dependency
        'torch',
        'ray',
        'tqdm',
        'ipdb',
        'torchvision',
        'scikit-learn',
        
        # Add all your dependencies here. These will be installed automatically when your package is installed.
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',  # Change as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Choose the appropriate license
        'Programming Language :: Python :: 3',  # Specify which py versions you support
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.9',  # Specify your project's Python version compatibility.
)
