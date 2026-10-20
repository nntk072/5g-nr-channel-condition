from sionna.phy.nr import PUSCHConfig

class SISOConfig:
    def __init__(self,
                 batch_size=1,
                 num_rx=1,
                 num_tx=1,
                 num_rx_ant=1,
                 num_tx_ant=1,
                 num_time_steps=14,
                 batch_size_cir=100,
                 num_prb=16,
                 mcs_index=14,
                 num_layers=1,
                 mcs_table=2,
                 domain="freq",
                 subcarrier_spacing=15e3,
                 num_symbols=14,
                 num_paths=12,
                 l_min=0,
                 min_speed=0,
                 max_speed=10
                 ):  # Add l_min parameter
        self.batch_size = batch_size
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant
        self.num_time_steps = num_time_steps
        self.batch_size_cir = batch_size_cir
        self.num_prb = num_prb
        self.mcs_index = mcs_index
        self.num_layers = num_layers
        self.mcs_table = mcs_table
        self.domain = domain
        self.subcarrier_spacing = subcarrier_spacing
        self.num_symbols = num_symbols
        self.num_time_steps = num_symbols
        self.num_paths = num_paths
        self.l_min = l_min  # Store l_min as an attribute
        
        self.min_speed = min_speed
        self.max_speed = max_speed
        
        # Initialize PUSCHConfig
        self.pusch_config = PUSCHConfig()
        self.pusch_config.carrier.subcarrier_spacing = self.subcarrier_spacing / 1000
        self.pusch_config.carrier.n_size_grid = self.num_prb
        self.pusch_config.num_antenna_ports = self.num_tx_ant
        self.pusch_config.num_layers = self.num_layers
        self.pusch_config.precoding = "non-codebook"
        self.pusch_config.tpmi = 1
        self.pusch_config.dmrs.dmrs_port_set = list(range(self.num_layers))
        self.pusch_config.dmrs.config_type = 1
        self.pusch_config.dmrs.length = 1
        self.pusch_config.dmrs.additional_position = 1
        self.pusch_config.dmrs.num_cdm_groups_without_data = 2
        self.pusch_config.dmrs.positions = [2, 11]  # DMRS at positions 2 and 11
        self.pusch_config.tb.mcs_index = self.mcs_index
        self.pusch_config.tb.mcs_table = self.mcs_table

    def create_pusch_configs(self):
        return [self.pusch_config]  # Only one config for SISO

    # Function for printing the config values
    def print_config(self):
        print(f"Number of PRBs: {self.num_prb}")
        print(f"MCS index: {self.mcs_index}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Number of antennas: {self.num_tx_ant}")
        print(f"Subcarrier spacing: {self.subcarrier_spacing}")
        print(f"Number of transmitters: {self.num_tx}")


class MIMOConfig:
    def __init__(self,
                batch_size=10,
                num_rx=1,
                num_tx=2,
                num_rx_ant=1,
                num_tx_ant=2,
                num_time_steps=14,
                num_prb=16,
                mcs_index=14,
                num_layers=1,
                mcs_table=2,
                domain="freq",
                subcarrier_spacing=15e3,
                num_symbols=14,
                num_paths=12,
                min_speed=0,
                max_speed=None,
                l_min=0):  # Add l_min parameter
        self.num_prb = num_prb
        self.mcs_index = mcs_index
        self.num_layers = num_layers
        self.mcs_table = mcs_table
        self.domain = domain
        self.num_tx_ant = num_tx_ant
        self.num_tx = num_tx
        self.subcarrier_spacing = subcarrier_spacing
        self.num_rx_ant = num_rx_ant
        self.num_rx = num_rx
        self.batch_size = batch_size
        self.num_time_steps = num_symbols
        self.num_paths = num_paths
        self.l_min = l_min  # Store l_min as an attribute
        self.min_speed = min_speed
        self.max_speed = max_speed
        # Initialize pusch_config here
        self.pusch_config = PUSCHConfig()
        self.pusch_config.carrier.subcarrier_spacing = self.subcarrier_spacing / 1000
        self.pusch_config.carrier.n_size_grid = self.num_prb
        self.pusch_config.num_antenna_ports = self.num_tx_ant
        self.pusch_config.num_layers = self.num_layers
        self.pusch_config.precoding = "codebook"
        self.pusch_config.tpmi = 1
        self.pusch_config.dmrs.dmrs_port_set = list(range(self.num_layers))
        self.pusch_config.dmrs.config_type = 1
        self.pusch_config.dmrs.length = 1
        self.pusch_config.dmrs.additional_position = 1
        self.pusch_config.dmrs.num_cdm_groups_without_data = 2
        self.pusch_config.tb.mcs_index = self.mcs_index
        self.pusch_config.tb.mcs_table = self.mcs_table

    def create_pusch_configs(self):
        pusch_configs = [self.pusch_config]  # Start with the base config
        for i in range(1, self.num_tx):
            pc = self.pusch_config.clone()  # Clone for each transmitter
            pc.dmrs.dmrs_port_set = list(
                range(i * self.num_layers, (i + 1) * self.num_layers))
            pusch_configs.append(pc)
        return pusch_configs

    # Function for printing the config values
    def print_config(self):
        print(f"Number of PRBs: {self.num_prb}")
        print(f"MCS index: {self.mcs_index}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Number of antennas: {self.num_tx_ant}")
        print(f"Subcarrier spacing: {self.subcarrier_spacing}")
        print(f"Number of transmitters: {self.num_tx}")