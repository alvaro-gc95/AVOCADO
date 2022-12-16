from ecmwfapi import ECMWFDataServer
import pandas as pd

server = ECMWFDataServer()


server.retrieve({
    'dataset': "era20c",
    'stream': "oper",
    'levtype': 'pl',
    'levelist': '500/850/925/1000',
    'time': "00/to/21",
    'date': "19000101",
    'step': "0",
    'type': "an",
    'param': "129",
    'target': "./data/geopotential.grib"
})
