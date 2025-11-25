from setuptools import setup, find_packages

setup(
    name='collaborative-multi-modal-agentic-framework',
    version='0.1.0',
    author='M Lucky',
    author_email='mlucky@gmail.com',
    description='A Collaborative Multi-Modal Agentic Framework for Business Intelligence',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',  # or 'torch' if using PyTorch
        'requests',
        'sqlalchemy',
        'pydantic',
        'pytest',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)