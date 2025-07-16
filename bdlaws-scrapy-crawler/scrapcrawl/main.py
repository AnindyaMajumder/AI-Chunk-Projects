import os
import sys
from scrapy import cmdline

# Change to the directory containing scrapy.cfg
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Execute the scrapy crawl command
cmdline.execute("scrapy crawl bdlaws".split())