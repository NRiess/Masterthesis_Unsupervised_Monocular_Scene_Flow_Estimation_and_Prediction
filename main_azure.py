from __future__ import absolute_import, division, print_function

import os


import subprocess
os.chdir("./models/correlation_package")
subprocess.check_call(["pip", "install", "."])
os.chdir("../forwardwarp_package")
subprocess.check_call(["pip", "install", "."])
os.chdir("../..")
os.makedirs('./runs', exist_ok=True)
print('installed modules')

import logging
import torch
from core import commandline, runtime, logger, tools, configuration as config
from main import main
main()
