# -*- coding:utf-8 -*-

import numpy as np
import math
from nr_prach_implementation import nr_prach_info
from nr_prach_implementation import nr_prach_seq
from common import nr_slot

class Prach():
    """ NR PRACH class"""
    def __init__(self, carrier_config, prach_config,prach_parameter):
        self.carrier_config = carrier_config
        self.prach_config = prach_config
        self.prach_parameter = prach_parameter

        carrier_scs = carrier_config['scs']
        BW = carrier_config['BW']
        self.carrier_prb_size = nr_slot.get_carrier_prb_size(carrier_scs, BW)

        self.prach_info = nr_prach_info.get_prach_config_info( \
            prach_config["prach_ConfigurationIndex"],carrier_config["duplex_type"])
        
        #select used PRACH format by the description at the end of 38.211 5.3.2 
        #If the preamble format given by Tables 6.3.3.2-2 to 6.3.3.2-4 is A1/B1, A2/B2 or A3/B3 ... 
        preamble_formats = self.prach_info["preamble_formats"]
        if len(preamble_formats) == 1:
            selected_format = preamble_formats[0]
        else:
            #for A1/B1, A2/B2 or A3/B3 case
            nRA_t = self.prach_parameter["nRA_t"]
            NRASlot_t = self.prach_info["NRASlot_t"]
            if nRA_t == NRASlot_t -1:
                selected_format = preamble_formats[1] #select B1, B2, B3
            else:
                selected_format = preamble_formats[0] #select A1, A2, A3
        self.selected_format = selected_format

        #get PRACH scs
        msg1_SubcarrierSpacing = self.prach_config["msg1_SubcarrierSpacing"]
        if selected_format in ['0', '1', '2']:
            msg1_SubcarrierSpacing = 1.25
        elif selected_format == '3':
            msg1_SubcarrierSpacing = 5
        self.msg1_SubcarrierSpacing = msg1_SubcarrierSpacing

        LRA, Nu, NRA_CP = nr_prach_info.get_prach_format_info(selected_format,msg1_SubcarrierSpacing)
        self.prach_info["LRA"] = LRA
        self.prach_info["Nu"] = Nu
        self.prach_info["NRA_CP"] = NRA_CP

        kbar,NRARB = nr_prach_info.get_kbar_NRARB(LRA, msg1_SubcarrierSpacing, carrier_scs)
        self.prach_info["kbar"] = kbar
        self.prach_info["NRARB"] = NRARB

        #calculate PRACh frequency offset Kk1+kbar
        #38.211 5.3.2 OFDM baseband signal generation for PRACH
        K = carrier_scs / msg1_SubcarrierSpacing
        nstart_RA =  prach_config['msg1_FrequencyStart']
        nRA = prach_parameter['nRA']
        assert nRA < prach_config['msg1_FDM']
        k1 = nstart_RA*12 + nRA*NRARB*12 - self.carrier_prb_size*12//2
        self.prach_freq_shift = K*k1 + kbar

        #calculate NRA_CP_L
        ActivePRACHslotinSubframe = prach_parameter['ActivePRACHslotinSubframe']
        start_symbol = self.prach_info["start_symbol"]
        nprachslot_insubframe = self.prach_info["nprachslot_insubframe"]
        NRA_dur = self.prach_info["NRA_dur"]
        nRA_t = self.prach_parameter["nRA_t"]

        nRA_slot, prach_first_symbol, NRA_CP_L,tRA_start = nr_prach_info.get_PRACH_txinfo(
            selected_format,ActivePRACHslotinSubframe,nRA_t,start_symbol,nprachslot_insubframe, \
            msg1_SubcarrierSpacing,Nu,NRA_CP,NRA_dur \
        )
        self.NRA_CP_L = NRA_CP_L
        self.prach_first_symbol = prach_first_symbol
        self.nRA_slot = nRA_slot
        self.tRA_start = tRA_start


    def process(self,sfn):
        """ finish PRACH processing, and 
        output:
        (1) 10ms timedomain waveform that put PRACH data into correspondent location
        (2) PRACH data which contain both CP and body
        (3) prach_active flag which indicate if this sfn contain PRACH data
        
        """
        #init waveform: 10ms, sample rate is fixed to 30.72Mhz
        samplerate_inkhz = 30720
        subframe_num = 10
        waveform = np.zeros(samplerate_inkhz*subframe_num,'c8')

        #check if the frame is PRACH active frame
        x = self.prach_info["x"]
        y = self.prach_info["y"]
        prach_data = []
        prach_active = 0
        if sfn % x != y:
            # this frame is not PRACH frame
            return waveform, prach_data,prach_active
        
        #
        prach_RootSequenceIndex = self.prach_config['prach_RootSequenceIndex']
        LRA = self.prach_info["LRA"]
        zeroCorrelationZoneConfig = self.prach_config['zeroCorrelationZoneConfig']
        PreambleIndex = self.prach_parameter['PreambleIndex']
        NRA_CP_L = self.NRA_CP_L
        Nu = self.prach_info["Nu"]
        tRA_start = self.tRA_start
        prach_freq_shift = self.prach_freq_shift
        msg1_SubcarrierSpacing = self.msg1_SubcarrierSpacing

        subframe_numbers = self.prach_info["subframe_numbers"]
        PRACH_subframe = self.prach_parameter['PRACH_subframe']

        if PRACH_subframe in subframe_numbers:
            prach_active = 1
            #generate PRACH sequence following to 38.211 6.3.3.1
            yuv = nr_prach_seq.PRACH_seq_gen(prach_RootSequenceIndex, LRA,zeroCorrelationZoneConfig,PreambleIndex)
            
            ifft_size = int(samplerate_inkhz/(msg1_SubcarrierSpacing)) 
            if LRA == 839:
                #support format 0,1,2 only
                repititions = Nu // 24576
                added_cp = _prach_preamble_process(yuv,ifft_size,samplerate_inkhz, \
                        Nu,prach_freq_shift,msg1_SubcarrierSpacing,repititions,NRA_CP_L,LRA)
            else:
                #LRA=139 
                if msg1_SubcarrierSpacing == 15:
                    repititions = Nu // 2048
                    added_cp = _prach_preamble_process(yuv,ifft_size,samplerate_inkhz, \
                            Nu,prach_freq_shift,msg1_SubcarrierSpacing,repititions,NRA_CP_L,LRA)
                else:
                    repititions = Nu // 1024
                    added_cp = _prach_preamble_process(yuv,ifft_size,samplerate_inkhz, \
                            Nu,prach_freq_shift,msg1_SubcarrierSpacing,repititions,NRA_CP_L,LRA)

            #put into waveform staring from correspondent subframe
            waveform[PRACH_subframe*samplerate_inkhz+tRA_start : PRACH_subframe*samplerate_inkhz+tRA_start+len(added_cp)] = added_cp

            #only output subframe data that contain full or partial PRACH data
            #samplerate_inkhz is number of sample in one subframe
            sel_datlen = math.ceil((tRA_start+len(added_cp))/samplerate_inkhz)*samplerate_inkhz
            #prach_data 
            prach_data = waveform[PRACH_subframe*samplerate_inkhz : PRACH_subframe*samplerate_inkhz+sel_datlen] #remove unnecessary zeors
        
        return waveform, prach_data, prach_active
    

def _prach_preamble_process(yuv,ifft_size,samplerate_inkhz,Nu,prach_freq_shift,msg1_SubcarrierSpacing,repititions,NRA_CP_L,LRA):
    """ PRACH IFFT, frequency shift, add CP
    """
    
    yuv_extend = np.concatenate([yuv, np.zeros(ifft_size-LRA)])
    td_yuv = np.fft.ifft(yuv_extend)
    #ifft total output power = total input power / IFFT_size
    #here compensate IFFT gain loss
    IFFT_gain = math.sqrt(ifft_size)
    td_yuv = td_yuv * IFFT_gain

    #frequency shift
    freq_comp = np.exp(1j*2*np.pi*prach_freq_shift*msg1_SubcarrierSpacing*np.arange(ifft_size)/samplerate_inkhz)
    td_yuv = td_yuv * freq_comp

    td_data = np.tile(td_yuv,repititions) #repete td_yuv repititions times
    added_cp = np.concatenate([td_data[-NRA_CP_L:],td_data])
    return added_cp

if __name__ == "__main__":
    print("test nr PRACH class and waveform generation")
    import test_nr_prach
    file_lists = test_nr_prach.get_nr_prach_testvector()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_nr_prach.test_nr_prach(filename)

    aaaa=1
    