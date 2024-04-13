from setuptools import setup, find_packages

setup(
    name='thepipe_api',
    version='0.1.0',
    author='Emmett McFarlane',
    author_email='emmett@thepi.pe',
    description='Automate information extraction for multimodal LLMs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/emcf/thepipe',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'thepipe=thepipe_api.thepipe:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True
)