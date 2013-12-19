# install Dependencie
SITE_PKG_DIR=$VIRTUAL_ENV/lib/python$TRAVIS_PYTHON_VERSION/site-packages
echo "Using SITE_PKG_DIR: $SITE_PKG_DIR"

# workaround for travis ignoring system_site_packages in travis.yml
rm -f $VIRTUAL_ENV/lib/python$TRAVIS_PYTHON_VERSION/no-global-site-packages.txt

sudo apt-get install libhdf5-serial-dev hdf5-tools python-tables
sudo apt-get install libatlas-base-dev liblapack-dev
sudo apt-get install python-numpy python-scipy python-matplotlib python-nose python-numexpr cython
sudo add-apt-repository -y ppa:pythonxy/pythonxy-devel && sudo apt-get -y update && sudo apt-get install -qq python-pandas
