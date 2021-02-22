from setuptools import setup, find_packages

setup(
    name = 'mergen',
    packages = find_packages(),
    version = '1.0',
    description = 'Unsupervised learning using data in time-domain astronomy', \
    author = 'Tansu Daylan, Emma Chickles, and Lindsey Gordon',
    author_email = 'tansu.daylan@gmail.com',
    url = 'https://github.com/tdaylan/mergen',
    download_url = 'https://github.com/tdaylan/mergen', 
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python'],
    #install_requires=['astrophy>=3'],
    include_package_data = True
    )

