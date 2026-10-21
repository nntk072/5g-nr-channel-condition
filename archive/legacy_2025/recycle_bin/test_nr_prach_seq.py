# -*- coding: utf-8 -*-
import pytest
from scipy import io
import os
import numpy as np
import math
from zipfile import ZipFile

from nr_prach_implementation import nr_prach_seq

def get_nr_prach_seq_testvectors():
    path = "tests/testvectors_prach_seq"
    if len(os.listdir(path)) < 100: #desn't unzip testvectors
        zipfile_lists = []
        for f in os.listdir(path):
            if f.endswith(".zip"):
                zipfile_lists.append(path + '/' + f)

        for zipfile in zipfile_lists:
            zObject = ZipFile(zipfile, 'r')
            zObject.extractall(path)

    file_lists = []
    for f in os.listdir(path):
        if f.endswith(".mat") and f.startswith("nrPRACH_sequence_testvec"):
            file_lists.append(path + '/' + f)
                        
    return file_lists

@pytest.mark.parametrize('filename', get_nr_prach_seq_testvectors())
def test_nr_prach_seq(filename):
    #read data
    matfile = io.loadmat(filename)
    yuv_ref = matfile["yuv"][0]
    prach_RootSequenceIndex = matfile["prach_RootSequenceIndex"][0][0]
    LRA = matfile["LRA"][0][0]
    zeroCorrelationZoneConfig = matfile["zeroCorrelationZoneConfig"][0][0]
    PreambleIndex = matfile["PreambleIndex"][0][0]

    yuv = nr_prach_seq.PRACH_seq_gen(prach_RootSequenceIndex, LRA,zeroCorrelationZoneConfig,PreambleIndex)
    assert np.allclose(yuv, yuv_ref,atol=1e-5)