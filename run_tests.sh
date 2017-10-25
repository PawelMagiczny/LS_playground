cd $(dirname "$0")
PYTHONPATH="`pwd`"
py.test -v methods/testlib/test_data_module.py

