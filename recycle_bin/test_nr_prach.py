# -*- coding: utf-8 -*-
import pytest
from scipy import io
import os
import numpy as np
import math
from zipfile import ZipFile

# from nr_prach import nr_prach
import nr_prach
from tests import read_prach_testvector

def get_nr_prach_testvector():
        
    skiptestveclist = ['nrPRACH_all_testvec_529_TDD__ConfigurationIndex_205.mat',
                       'nrPRACH_all_testvec_1307_TDD__ConfigurationIndex_209.mat',
                       'nrPRACH_all_testvec_1291_TDD__ConfigurationIndex_205.mat',
                       'nrPRACH_all_testvec_1285_TDD__ConfigurationIndex_204.mat',
                       'nrPRACH_all_testvec_523_TDD__ConfigurationIndex_204.mat',
                       'nrPRACH_all_testvec_511_TDD__ConfigurationIndex_201.mat',
                       'nrPRACH_all_testvec_519_TDD__ConfigurationIndex_203.mat',
                       'nrPRACH_all_testvec_1281_TDD__ConfigurationIndex_203.mat',
                       'nrPRACH_all_testvec_1297_TDD__ConfigurationIndex_207.mat',
                       'nrPRACH_all_testvec_535_TDD__ConfigurationIndex_207.mat',
                       'nrPRACH_all_testvec_539_TDD__ConfigurationIndex_208.mat',
                       'nrPRACH_all_testvec_1273_TDD__ConfigurationIndex_201.mat',
                       'nrPRACH_all_testvec_545_TDD__ConfigurationIndex_209.mat',
                       'nrPRACH_all_testvec_1301_TDD__ConfigurationIndex_208.mat',
                       ]
    path = "tests/testvectors_prach"
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
        if f.endswith(".mat") and f.startswith("nrPRACH_all_testvec"):
            if f in skiptestveclist:
                continue
            file_lists.append(path + '/' + f)
            
    return file_lists

@pytest.mark.parametrize('filename', get_nr_prach_testvector())
def test_nr_prach(filename):
    sfn = 0 #matlab 5g toolbox nr_prach module onlys upport sfn=0 case

    matfile = io.loadmat(filename)
    waveform_ref, prachSymbols_out, prachGrid_out, prachtv_cfg = read_prach_testvector.read_nrPRACH_all_testvec_matfile(matfile)
    carrier_config, prach_config, prach_parameters = read_prach_testvector.gen_prach_alltestvec_config(prachtv_cfg)

    nrprach = nr_prach.Prach(carrier_config, prach_config, prach_parameters)
    waveform, prach_data, prach_active = nrprach.process(sfn)

    #compare
    #matlab generated prach add additiona 1/sqrt(Nfft*LRA) gain
    if carrier_config['scs'] == 30:
        carrier_Nfft = 1024
    else:
        carrier_Nfft = 2048

    LRA = prachtv_cfg['LRA']
            
    if prachtv_cfg['ActivePRACHSlot'] == 0:
        sel_waveform_ref = waveform_ref[0:waveform.shape[0]]
        assert np.allclose(prach_data, sel_waveform_ref*math.sqrt(carrier_Nfft*LRA),atol=1e-5)
    else:
        sel_waveform = prach_data[15360:]
        sel_waveform_ref = sel_waveform[0:waveform.shape[0]]
        assert np.allclose(sel_waveform, sel_waveform_ref,atol=1e-5)

