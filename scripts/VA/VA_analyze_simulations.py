"""
load and analyze some of the simulation data.
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

project_path = "/home/nick/Documents/Summer 2022 (C4GC with BII)/measles_metapop/{}"
sys.path.append(project_path.format("scripts/"))

from spatial_tsir import *

#%%

pool = spatial_tSIR_pool(load=project_path.format("outputs/va_sensitivity_analysis/outflux_11_6_VA_analysis.save"))
