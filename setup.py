from setuptools import setup, find_packages

def read_requirements(file):
    with open(file, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('git+')]

def read_git_requirements(file):
    with open(file, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip().startswith('git+')]

setup(
    name='thepipe_api',
    version='1.2.8',
    author='Emmett McFarlane',
    author_email='emmett@thepi.pe',
    description='AI-native extractor, powered by multimodal LLMs.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/emcf/thepipe',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=read_requirements('requirements.txt'),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'thepipe=thepipe.__init__:main',
        ],
    },
    extras_require={
        'local': read_requirements('local.txt'),
    },
    dependency_links=read_git_requirements('local.txt')
)
