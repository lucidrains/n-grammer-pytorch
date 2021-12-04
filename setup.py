from setuptools import setup, find_packages

setup(
  name = 'n-grammer-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.10',
  license='MIT',
  description = 'N-Grammer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/n-grammer-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'n-grams',
    'memory'
  ],
  install_requires=[
    'einops>=0.3',
    'sympy',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
