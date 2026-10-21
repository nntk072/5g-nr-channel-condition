# SISO/MIMO/SIMO configs
# from nr_pusch.configs import SISOConfig, MIMOConfig

from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import KBestDetector, LinearDetector
from sionna.phy.utils import compute_ber, ebnodb2no
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.channel import OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.phy.channel.tr38901 import CDL, TDL
from sionna.phy.channel.tr38901 import AntennaArray, Antenna
import tensorflow as tf
import numpy as np

# Import Sionna RT components


class PUSCHModel(tf.keras.Model):
    def __init__(self, h_freq, ebno_db=2.0, batch_size=100):
        super().__init__()
        self.ebno_db = ebno_db
        self.batch_size = batch_size
        self.h_freq = h_freq  # Store h_freq as attribute
        # Simplified PUSCHConfig
        self.pusch_config = PUSCHConfig()
        self.pusch_transmitter = PUSCHTransmitter(self.pusch_config)
        self.pusch_receiver = PUSCHReceiver(self.pusch_transmitter)
        self.no = self.compute_noise()
        self.channel = ApplyOFDMChannel(add_awgn=True)

    def compute_noise(self):
        return ebnodb2no(
            self.ebno_db,
            self.pusch_transmitter._num_bits_per_symbol,
            self.pusch_transmitter._target_coderate,
            self.pusch_transmitter.resource_grid,
        )

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        x, b = self.pusch_transmitter(batch_size)
        y = self.channel([x, self.h_freq, self.no])
        b_hat = self.pusch_receiver([y, self.no])
        return b, b_hat

    def run(self):
        b, b_hat = self.call(self.batch_size, self.ebno_db)
        ber = compute_ber(b, b_hat).numpy()
        print(f"BER: {ber:.5f}")


class PUSCHReceiverExtension:
    def __init__(
        self, config, detector, perfect_csi, transmitter
    ):  # Add transmitter as argument
        self.config = config
        self.perfect_csi = perfect_csi
        rx_tx_association = np.ones([1, config.num_tx], bool)
        stream_management = StreamManagement(rx_tx_association, config.num_layers)

        assert detector in [None, "lmmse", "kbest"], "Unsupported MIMO detector"
        if detector == None:
            self.detector = None
        elif detector == "lmmse":
            self.detector = LinearDetector(
                equalizer="lmmse",
                output="bit",
                demapping_method="maxlog",
                resource_grid=transmitter.pusch_transmitter.resource_grid,  # Use transmitter here
                stream_management=stream_management,
                constellation_type="qam",
                num_bits_per_symbol=config.create_pusch_configs()[
                    0
                ].tb.num_bits_per_symbol,
            )
        elif detector == "kbest":
            self.detector = KBestDetector(
                output="bit",
                num_streams=config.num_tx * config.num_layers,
                k=64,
                resource_grid=transmitter.pusch_transmitter.resource_grid,  # Use transmitter here
                stream_management=stream_management,
                constellation_type="qam",
                num_bits_per_symbol=config.create_pusch_configs()[
                    0
                ].tb.num_bits_per_symbol,
            )

        if self.perfect_csi:
            self.pusch_receiver = PUSCHReceiver(
                transmitter.pusch_transmitter,  # Use transmitter here
                mimo_detector=self.detector,
                input_domain=config.domain,
                channel_estimator="perfect",
                l_min=config.l_min if config.domain == "time" else None,
            )
        else:
            self.pusch_receiver = PUSCHReceiver(
                transmitter.pusch_transmitter,  # Use transmitter here
                mimo_detector=self.detector,
                input_domain=config.domain,
                l_min=config.l_min if config.domain == "time" else None,
            )


class Channel:
    def __init__(
        self,
        config,
        a,
        tau,
        model_type="",
        scenario="",
        channel_model=None,
    ):
        self.a = a
        self.tau = tau
        resource_grid = PUSCHTransmitterExtension(
            config
        ).pusch_transmitter.resource_grid

        # Check if model_type is valid
        assert model_type.lower() in ["cdl", "tdl", "sionna"], "Invalid model type"
        # self.channel_model = None
        if channel_model is None:
            if model_type.lower() == "cdl":
                # carrier_frequency = 2.14e9  # Carrier frequency in Hz.
                carrier_frequency = 3.5e9  # Carrier frequency in Hz.
                # This is needed here to define the antenna element spacing
                # Provide the missing arguments to PanelArray
                num_ut_ant = config.num_tx_ant
                num_bs_ant = config.num_rx_ant
                # Print the number of antennas
                # check if num_ut_ant and num_bs_ant are 1 or not
                if num_ut_ant != 1 and num_bs_ant != 1:
                    ut_array = AntennaArray(
                        num_rows=1,
                        num_cols=int(num_ut_ant / 2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency,
                    )
                    bs_array = AntennaArray(
                        num_rows=1,
                        num_cols=int(num_bs_ant / 2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency,
                    )
                else:
                    ut_array = AntennaArray(
                        num_rows=1,
                        num_cols=1,
                        antenna_pattern="38.901",
                        polarization="single",
                        polarization_type="V",
                        carrier_frequency=carrier_frequency,
                    )
                    bs_array = AntennaArray(
                        num_rows=1,
                        num_cols=1,
                        antenna_pattern="38.901",  # UE orientation is random
                        polarization="single",
                        polarization_type="V",
                        carrier_frequency=carrier_frequency,
                    )
                # Check if the scenario is valid, if not, setting the default as A with notice
                if scenario.upper() not in ["A", "B", "C", "D", "E"]:
                    print("Invalid scenario, defaulting to A")
                    scenario = "A"
                # Initialize the CDL model with the antenna arrays
                
                self.channel_model = CDL(
                    model=scenario.upper(),
                    ut_array=ut_array,  # Add ut_array
                    bs_array=bs_array,  # Add bs_array
                    delay_spread=300e-9, # rural,
                    carrier_frequency=3.5e9,
                    # direction="uplink",
                    direction="downlink",
                    min_speed=config.min_speed,
                    max_speed=config.max_speed,
                    # dopplershift = 0, delay spread 
                )
                self.a, self.tau = self.channel_model(
                    batch_size=config.batch_size,
                    num_time_steps=config.num_time_steps,
                    sampling_frequency=config.subcarrier_spacing,
                )
                print(f"CDL model initialized with scenario: {scenario}")
                print(f"Shape of a: {self.a.shape}")
                print(f"Shape of tau: {self.tau.shape}")
                print(f"Shape of a: {self.a.shape}")
                print(f"Shape of tau: {self.tau.shape}")
                # cir_generator = CIRGenerator(self.a, self.tau, config.num_tx, scenario)
                # self.channel_model = CIRDataset(
                #     cir_generator,
                #     config.batch_size,
                #     config.num_rx,
                #     config.num_rx_ant,
                #     config.num_tx,
                #     config.num_tx_ant,
                #     config.num_paths,
                #     config.num_time_steps,
                # )

                # print(f"ChannelModel: {self.channel_model.shape()}")
            elif model_type.lower() == "tdl":
                if scenario.upper() not in ["A", "B", "C", "D", "E"]:
                    print("Invalid scenario, defaulting to A")
                    scenario = "A"

                self.channel_model = TDL(
                    model=scenario.upper(),
                    num_rx_ant=config.num_rx_ant,
                    num_tx_ant=config.num_tx_ant,
                    delay_spread=300e-9,
                    carrier_frequency=3.5e9,
                    min_speed=config.min_speed,
                    max_speed=config.max_speed,
                )
                self.a, self.tau = self.channel_model(
                    batch_size=config.batch_size,
                    num_time_steps=config.num_time_steps,
                    sampling_frequency=config.subcarrier_spacing,
                )
                # print(f"TDL model initialized with scenario: {scenario}")
                # print(f"Shape of a: {self.a.shape}")
                # print(f"Shape of tau: {self.tau.shape}")
                # cir_generator = CIRGenerator(self.a, self.tau, config.num_tx, scenario)

                # self.channel_model = CIRDataset(
                #     cir_generator,
                #     config.batch_size,
                #     config.num_rx,
                #     config.num_rx_ant,
                #     config.num_tx,
                #     config.num_tx_ant,
                #     config.num_paths,
                #     config.num_time_steps,
                # )

            elif model_type.lower() == "sionna":
                # Init CIR generator
                cir_generator = CIRGenerator(a, tau, config.num_tx, scenario)

                # Initialises a channel model that can be directly used by OFDMChannel layer
                self.channel_model = CIRDataset(
                    cir_generator,
                    config.batch_size,
                    config.num_rx,
                    config.num_rx_ant,
                    config.num_tx,
                    config.num_tx_ant,
                    # 12,
                    config.num_paths,
                    config.num_time_steps,
                )
        else:
            self.channel_model = channel_model
        self._channel = OFDMChannel(
            self.channel_model,
            resource_grid,
            normalize_channel=True,
            return_channel=True,
        )
        
    def __call__(self, inputs):
        y, self.h = self._channel(inputs)  # Store the CIR in self.h
        return y, self.h

    def return_a_tau(self):
        return self.a, self.tau


class CIRGenerator:
    """Creates a generator from a given dataset of channel impulse responses.

    The generator samples ``num_tx`` different transmitters from the given path
    coefficients `a` and path delays `tau` and stacks the CIRs into a single tensor.

    Note that the generator internally samples ``num_tx`` random transmitters
    from the dataset. For this, the inputs ``a`` and ``tau`` must be given for
    a single transmitter (i.e., ``num_tx`` =1) which will then be stacked
    internally.

    Parameters
    ----------
    a : [batch size, num_rx, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps], complex
        Path coefficients per transmitter.

    tau : [batch size, num_rx, 1, num_paths], float
        Path delays [s] per transmitter.

    num_tx : int
        Number of transmitters

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    def __init__(self, a, tau, num_tx, scenario="siso"):

        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]

        self._num_tx = num_tx

        self._scenario = scenario

    def __call__(self):

        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample 4 random users and stack them together
            idx, _, _ = tf.random.uniform_candidate_sampler(
                tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                num_true=self._dataset_size,
                num_sampled=self._num_tx,
                unique=True,
                range_max=self._dataset_size,
            )

            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to remove batch dimension
            # if self._scenario.lower() == "mimo":
            a = tf.transpose(a, (3, 1, 2, 0, 4, 5, 6))
            tau = tf.transpose(tau, (2, 1, 0, 3))
            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)
            # print(f"Shape of a after going through CIRGenerator: {a.shape}")
            # print(f"Shape of tau after going through CIRGenerator: {tau.shape}")
            yield a, tau


class PUSCHTransmitterExtension:
    def __init__(self, config):
        self.config = config
        self.pusch_transmitter = PUSCHTransmitter(
            config.create_pusch_configs(), output_domain=config.domain
        )


class ChannelModel:
    def __init__(
        self,
        cir_generator,
        batch_size,
        num_rx,
        num_rx_ant,
        num_tx,
        num_tx_ant,
        num_paths,
        num_time_steps,
    ):
        self.cir_generator = cir_generator
        self.batch_size = batch_size
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_paths = num_paths
        self.num_time_steps = num_time_steps

    # def __call__(self):
    def __call__(self):
        return self.cir_generator()

    def __len__(self):
        return self.batch_size


class Model(tf.keras.Model):
    def __init__(
        self,
        perfect_csi,
        detector,
        config,
        model_type="",
        scenario="",
        a=None,
        tau=None,
        paths=None,
        channel_model=None,
    ):
        super().__init__()
        self.config = config
        self.transmitter = PUSCHTransmitterExtension(self.config)
        self.receiver = PUSCHReceiverExtension(
            self.config, detector, perfect_csi, self.transmitter
        )
        
        self.a = a
        self.tau = tau
        if channel_model is not None:
            self.channel = Channel(
                self.config,
                self.a,
                self.tau,
                model_type,
                scenario,
                channel_model=channel_model,
            )
        else:
            self.channel = Channel(
                self.config,
                self.a,
                self.tau,
                model_type,
                scenario,
            )
        if a is None:
            # returns a, tau from channel model
            self.a, self.tau = self.channel.return_a_tau()
        self.model_type = model_type
        self.paths = paths



    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        self.x, self.b = self.transmitter.pusch_transmitter(self.config.batch_size)
        no = ebnodb2no(
            ebno_db,
            self.transmitter.pusch_transmitter._num_bits_per_symbol,
            self.transmitter.pusch_transmitter._target_coderate,
            self.transmitter.pusch_transmitter.resource_grid,
        )

        y, h = self.channel._channel([self.x, no])

        if self.receiver.perfect_csi:
            b_hat = self.receiver.pusch_receiver([y, h, no])
        else:
            b_hat = self.receiver.pusch_receiver([y, no])
        return self.b, b_hat

    @tf.function(jit_compile=False)
    def run(self, ebno_db):

        b, b_hat = self.call(self.config.batch_size, ebno_db)

        ber = compute_ber(b, b_hat)
        print(ber)
        print(f"BER: {ber.numpy()}")

    # @tf.function(jit_compile=False)
    # @tf.function
    def extract_features(self, ebno_db):
        paths = self.paths
        self.x, self.b = self.transmitter.pusch_transmitter(self.config.batch_size)
        no = ebnodb2no(
            ebno_db,
            self.transmitter.pusch_transmitter._num_bits_per_symbol,
            self.transmitter.pusch_transmitter._target_coderate,
            self.transmitter.pusch_transmitter.resource_grid,
        )
        # print(self.x.shape)
        y, h = self.channel([self.x, no])

        # Signal Power
        signal_power = tf.reduce_mean(tf.abs(y) ** 2)
        noise_power = tf.reduce_mean(tf.abs(no) ** 2)

        # Highest Power
        highest_power = tf.reduce_max(tf.abs(y) ** 2)

        # Kurtosis
        power = tf.abs(y) ** 2
        mean_power = tf.reduce_mean(power)
        kurtosis = tf.reduce_mean((power - mean_power) ** 4) / (
            tf.reduce_mean((power - mean_power) ** 2) ** 2
        )

        # Skewness
        skewness = tf.reduce_mean((power - mean_power) ** 3) / (
            tf.reduce_mean((power - mean_power) ** 2) ** 1.5
        )

        # CIR realizations
        t = self.tau[0, 0, 0, :] / 1e-9  # Scale to ns
        a_abs = np.abs(self.a)[0, 0, 0, 0, 0, :, 0] # This is beforee the channel model
        # print(a_abs.shape) # Shape description: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # print(h.shape)     # Shape description: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps]
        # print(y.shape)
        # plt.figure()
        # plt.title("Channel impulse response realization")
        # plt.stem(t, a_abs)
        # plt.xlabel(r"$\tau$ [ns]")
        # plt.ylabel(r"$|a|$")
        # plt.show()

        # plt.figure()
        # plt.title("Channel impulse response realization after channel model")
        # plt.stem(t, h_abs)
        # plt.xlabel(r"$\tau$ [ns]")
        # plt.ylabel(r"$|a|$")
        # plt.show()

        if self.model_type.lower() == "sionna":
            power = tf.abs(y) ** 2

            aoa = paths.theta_r
            aod = paths.theta_t
            zod = paths.phi_r
            zoa = paths.phi_t
            toa = self.tau

            # print(toa.shape)
            # print(power.shape)
            # Rising time is the estimation of the time period between the first MPC and the strongest MPC
            first_mpc_time = tf.reduce_min(toa)
            strongest_mpc_time = tf.gather(toa, tf.argmax(power))
            rising_time = strongest_mpc_time - first_mpc_time

            # RMS Delay Spread
            # excess_mean_delay = tf.reduce_sum(toa_broadcasted * power_expanded) / tf.reduce_sum(power_expanded)
            # rms_delay_spread = tf.sqrt(tf.reduce_sum((toa_broadcasted - excess_mean_delay) ** 2 * power_expanded) / tf.reduce_sum(power_expanded))

            # Delay Spread
            mean_delay = tf.reduce_mean(paths.tau)
            delay_spread = tf.sqrt(tf.reduce_mean(tf.square(paths.tau - mean_delay)))

            # # Rician K-factor
            # los_mask = tf.expand_dims(paths.types == Paths.LOS, axis=-1)  # Expand dims to match power shape
            # nlos_mask = tf.expand_dims(~los_mask, axis=-1)  # Expand dims to match power shape
            # print(los_mask.shape)
            # print(nlos_mask.shape)
            # print(power.shape)
            # P_LOS = tf.reduce_sum(tf.abs(self.a) ** 2 * tf.cast(los_mask, tf.float32))
            # P_NLOS = tf.reduce_sum(tf.abs(paths.a) ** 2 * tf.cast(nlos_mask, tf.float32))

            # k_factor = P_LOS / P_NLOS if P_NLOS > 0 else tf.constant(0.0, dtype=tf.float32)
            k_factor = None
            # angular_spread = tf.reduce_mean(tf.abs(paths.a))
            # calculate angular spread based on the formula
            angular_spread = tf.reduce_mean(
                tf.sqrt(
                    (paths.theta_r - tf.reduce_mean(paths.theta_r)) ** 2
                    + (paths.phi_r - tf.reduce_mean(paths.phi_r)) ** 2
                )
            )

            # Mean direction of the Power angular spectrum
            power = tf.reshape(power, [-1])  # Flatten the power tensor if necessary
            theta_r = tf.reshape(
                paths.theta_r, [-1]
            )  # Flatten the theta_r tensor if necessary
            phi_r = tf.reshape(
                paths.phi_r, [-1]
            )  # Flatten the phi_r tensor if necessary

            min_length = min(
                tf.shape(power)[0], tf.shape(theta_r)[0], tf.shape(phi_r)[0]
            )
            power = power[:min_length]
            theta_r = theta_r[:min_length]
            phi_r = phi_r[:min_length]

            mean_theta_real = tf.reduce_sum(tf.cos(theta_r) * power) / tf.reduce_sum(
                power
            )
            mean_theta_imag = tf.reduce_sum(tf.sin(theta_r) * power) / tf.reduce_sum(
                power
            )
            mean_theta = tf.atan2(mean_theta_imag, mean_theta_real)

            mean_phi_real = tf.reduce_sum(tf.cos(phi_r) * power) / tf.reduce_sum(power)
            mean_phi_imag = tf.reduce_sum(tf.sin(phi_r) * power) / tf.reduce_sum(power)
            mean_phi = tf.atan2(mean_phi_imag, mean_phi_real)

            # Angular difference
            # angular_difference = tf.reduce_mean(
            #    tf.abs(tf.exp(1j * paths.theta_t) - tf.exp(1j * paths.theta_r))
            # )
            angular_difference = tf.reduce_mean(
                tf.abs(
                    tf.complex(tf.cos(paths.theta_t), tf.sin(paths.theta_t))
                    - tf.complex(tf.cos(paths.theta_r), tf.sin(paths.theta_r))
                )
            )

            # Ensure the shapes are compatible for matrix multiplication
            theta_r_reshaped = tf.reshape(paths.theta_r, [-1, 1])
            theta_t_reshaped = tf.reshape(paths.theta_t, [-1, 1])
            angular_variant = tf.reduce_sum(
                tf.linalg.eigvals(
                    tf.stack(
                        [
                            tf.math.real(
                                tf.linalg.matmul(
                                    tf.math.conj(theta_r_reshaped),
                                    theta_r_reshaped,
                                    transpose_a=True,
                                )
                            ),
                            tf.math.real(
                                tf.linalg.matmul(
                                    tf.math.conj(theta_r_reshaped),
                                    theta_t_reshaped,
                                    transpose_a=True,
                                )
                            ),
                            tf.math.real(
                                tf.linalg.matmul(
                                    tf.math.conj(theta_t_reshaped),
                                    theta_r_reshaped,
                                    transpose_a=True,
                                )
                            ),
                            tf.math.real(
                                tf.linalg.matmul(
                                    tf.math.conj(theta_t_reshaped),
                                    theta_t_reshaped,
                                    transpose_a=True,
                                )
                            ),
                        ]
                    )
                )
            )

            doppler_shift = paths.doppler
        elif self.model_type.lower() == "tdl":
            aoa = None  # TDL model does not provide AoA
            aod = None  # TDL model does not provide AoD
            zod = None  # TDL model does not provide ZoD
            zoa = None  # TDL model does not provide ZoA
            toa = self.channel.channel_model.delays
            # Rising time is the estimation of the time period between the first MPC and the strongest MPC
            first_mpc_time = tf.reduce_min(toa)
            strongest_mpc_time = tf.gather(toa, tf.argmax(power))
            rising_time = strongest_mpc_time - first_mpc_time

            # rising_time = tf.reduce_mean(self.channel.channel_model.delays)
            delay_spread = self.channel.channel_model.delay_spread

            # # # RMS Delay Spread
            # excess_mean_delay = tf.reduce_sum(tf.expand_dims(toa, axis=-1) * power) / tf.reduce_sum(power)
            # rms_delay_spread = tf.sqrt(tf.reduce_sum((tf.expand_dims(toa, axis=-1) - excess_mean_delay) ** 2 * power) / tf.reduce_sum(power))

            if self.channel.channel_model.los:
                k_factor = self.channel.channel_model.k_factor
            else:
                k_factor = None

            # Calculate Doppler shift
            batch_size = self.config.batch_size
            v_r = tf.random.uniform(
                shape=[batch_size, 1],
                minval=self.channel.channel_model._min_speed,
                maxval=self.channel.channel_model._max_speed,
                dtype=self.channel.channel_model._real_dtype,
            )
            v_phi = tf.random.uniform(
                shape=[batch_size, 1],
                minval=0.0,
                maxval=2.0 * np.pi,
                dtype=self.channel.channel_model._real_dtype,
            )
            v_theta = tf.random.uniform(
                shape=[batch_size, 1],
                minval=0.0,
                maxval=np.pi,
                dtype=self.channel.channel_model._real_dtype,
            )
            velocities = tf.stack(
                [
                    v_r * tf.cos(v_phi) * tf.sin(v_theta),
                    v_r * tf.sin(v_phi) * tf.sin(v_theta),
                    v_r * tf.cos(v_theta),
                ],
                axis=-1,
            )
            doppler_shift = tf.reduce_mean(tf.abs(velocities))

            # TDL model does not provide angular spread
            angular_spread = None
            mean_theta = None
            mean_phi = None
            angular_difference = None
            angular_variant = None
        elif self.model_type.lower() == "cdl":
            aoa = self.channel.channel_model._aoa
            aod = self.channel.channel_model._aod
            zod = self.channel.channel_model._zod
            zoa = self.channel.channel_model._zoa
            toa = self.channel.channel_model.delays
            # Rising time is the estimation of the time period between the first MPC and the strongest MPC
            first_mpc_time = tf.reduce_min(toa)
            strongest_mpc_time = tf.gather(toa, tf.argmax(power))
            rising_time = strongest_mpc_time - first_mpc_time

            # # # RMS Delay Spread
            # excess_mean_delay = tf.reduce_sum(tf.expand_dims(toa, axis=-1) * power) / tf.reduce_sum(power)
            # rms_delay_spread = tf.sqrt(tf.reduce_sum((tf.expand_dims(toa, axis=-1) - excess_mean_delay) ** 2 * power) / tf.reduce_sum(power))

            # Delay Spread
            delay_spread = self.channel.channel_model.delay_spread

            if self.channel.channel_model.los:
                k_factor = self.channel.channel_model.k_factor
            else:
                k_factor = None

            angular_spread = (
                tf.reduce_mean(aoa)
                + tf.reduce_mean(aod)
                + tf.reduce_mean(zod)
                + tf.reduce_mean(zoa)
            ) / 4


            # Mean direction of the Power angular spectrum
            power = tf.reshape(power, [-1])  # Flatten the power tensor if necessary
            theta_r = tf.reshape(aoa, [-1])
            phi_r = tf.reshape(zoa, [-1])

            min_length = min(
                tf.shape(power)[0], tf.shape(theta_r)[0], tf.shape(phi_r)[0]
            )
            power = power[:min_length]
            theta_r = theta_r[:min_length]
            phi_r = phi_r[:min_length]

            mean_theta_real = tf.reduce_sum(tf.cos(theta_r) * power) / tf.reduce_sum(
                power
            )
            mean_theta_imag = tf.reduce_sum(tf.sin(theta_r) * power) / tf.reduce_sum(
                power
            )
            mean_theta = tf.atan2(mean_theta_imag, mean_theta_real)

            mean_phi_real = tf.reduce_sum(tf.cos(phi_r) * power) / tf.reduce_sum(power)
            mean_phi_imag = tf.reduce_sum(tf.sin(phi_r) * power) / tf.reduce_sum(power)
            mean_phi = tf.atan2(mean_phi_imag, mean_phi_real)

            angular_difference = tf.reduce_mean(
                tf.abs(
                    tf.complex(tf.cos(aod), tf.sin(aod))
                    - tf.complex(tf.cos(aoa), tf.sin(aoa))
                )
            )

            # Ensure the shapes are compatible for matrix multiplication
            theta_r_reshaped = tf.reshape(aoa, [-1, 1])
            theta_t_reshaped = tf.reshape(aod, [-1, 1])
            angular_variant = tf.reduce_sum(
                tf.linalg.eigvals(
                    tf.stack(
                        [
                            tf.math.real(
                                tf.linalg.matmul(
                                    tf.math.conj(theta_r_reshaped),
                                    theta_r_reshaped,
                                    transpose_a=True,
                                )
                            ),
                            tf.math.real(
                                tf.linalg.matmul(
                                    tf.math.conj(theta_r_reshaped),
                                    theta_t_reshaped,
                                    transpose_a=True,
                                )
                            ),
                            tf.math.real(
                                tf.linalg.matmul(
                                    tf.math.conj(theta_t_reshaped),
                                    theta_r_reshaped,
                                    transpose_a=True,
                                )
                            ),
                            tf.math.real(
                                tf.linalg.matmul(
                                    tf.math.conj(theta_t_reshaped),
                                    theta_t_reshaped,
                                    transpose_a=True,
                                )
                            ),
                        ]
                    )
                )
            )

            batch_size = self.config.batch_size
            v_r = tf.random.uniform(
                shape=[batch_size, 1],
                minval=self.channel.channel_model._min_speed,
                maxval=self.channel.channel_model._max_speed,
                dtype=self.channel.channel_model._real_dtype,
            )
            v_phi = tf.random.uniform(
                shape=[batch_size, 1],
                minval=0.0,
                maxval=2.0 * np.pi,
                dtype=self.channel.channel_model._real_dtype,
            )
            v_theta = tf.random.uniform(
                shape=[batch_size, 1],
                minval=0.0,
                maxval=np.pi,
                dtype=self.channel.channel_model._real_dtype,
            )
            velocities = tf.stack(
                [
                    v_r * tf.cos(v_phi) * tf.sin(v_theta),
                    v_r * tf.sin(v_phi) * tf.sin(v_theta),
                    v_r * tf.cos(v_theta),
                ],
                axis=-1,
            )
            doppler_shift = tf.reduce_mean(tf.abs(velocities))
        features = {
            "y": safe_numpy(y),
            "h": safe_numpy(h),
            "a_abs": safe_numpy(a_abs),
            "t": safe_numpy(t),
            "snr": safe_numpy(
                10 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)
            ),
            "highest_received_power": safe_numpy(highest_power),
            "kurtosis": safe_numpy(kurtosis),
            "skewness": safe_numpy(skewness),
            "mean": safe_numpy(tf.reduce_mean(y)),
            "std": safe_numpy(tf.math.reduce_std(y)),
            "aoa": safe_numpy(aoa) if aoa is not None else aoa,
            "aod": safe_numpy(aod) if aod is not None else aod,
            "zoa": safe_numpy(zoa) if zoa is not None else zoa,
            "zod": safe_numpy(zod) if zod is not None else zod,
            "toa_prs": safe_numpy(toa),
            "rising_time": safe_numpy(rising_time),
            "delay_spread": safe_numpy(delay_spread),
            # "rms_delay_spread": safe_numpy(rms_delay_spread),
            "rician_k_factor": safe_numpy(k_factor),
            "angular_spread": (
                safe_numpy(angular_spread)
                if angular_spread is not None
                else angular_spread
            ),
            "mean_theta": (
                safe_numpy(mean_theta) if mean_theta is not None else mean_theta
            ),
            "mean_phi": safe_numpy(mean_phi) if mean_phi is not None else mean_phi,
            "angular_difference": (
                safe_numpy(angular_difference)
                if angular_difference is not None
                else angular_difference
            ),
            "angular_variant": (
                safe_numpy(angular_variant)
                if angular_variant is not None
                else angular_variant
            ),
            "doppler_shift": safe_numpy(doppler_shift),
            # "normalized_delay_power_correlation": safe_numpy(
            #     self.compute_normalized_delay_power_correlation(y, h)
            # ),
        }
        return features

    # @tf.function(jit_compile=False)
    def compute_normalized_delay_power_correlation(self, y, h):
        if self.model_type.lower() in ["cdl", "tdl"]:
            # For CDL and TDL models, use the channel model directly
            # print("Model type is CDL/TDL")
            # print(y.shape)
            # print(h.shape)
            y_tiled = tf.reshape(y, [-1])
            h_tiled = tf.reshape(h, [-1])
            # print(y_tiled.shape)
            # print(h_tiled.shape)
            # Ensure the shapes are compatible for broadcasting
            if y_tiled.shape[0] < h_tiled.shape[0]:
                h_tiled = h_tiled[: y_tiled.shape[0]]
            elif y_tiled.shape[0] > h_tiled.shape[0]:
                y_tiled = y_tiled[: h_tiled.shape[0]]
            correlation = tf.reduce_mean(y_tiled * tf.math.conj(h_tiled))
            return correlation.numpy()

        # For Sionna model
        # print("Model type is not CDL/TDL")
        y_expanded = y
        y_tiled = y_expanded
        # Check if SISO or MIMO to expand
        if self.config.num_tx != 1 or self.config.num_tx_ant != 1:
            if self.config.num_tx != 1:
                y_expanded = tf.expand_dims(y, axis=self.config.num_tx - 1)
            if self.config.num_tx_ant != 1:
                y_expanded = tf.expand_dims(y_expanded, axis=self.config.num_tx_ant)
                y_tiled = tf.tile(
                    y_expanded,
                    [1, 1, 1, self.config.num_tx, self.config.num_tx_ant, 1, 1],
                )
        else:
            y_tiled = y

        y_tiled = tf.reshape(y_tiled, [-1])
        h_tiled = tf.reshape(h, [-1])

        # Ensure the shapes are compatible for broadcasting
        if y_tiled.shape[0] < h_tiled.shape[0]:
            h_tiled = h_tiled[: y_tiled.shape[0]]
        elif y_tiled.shape[0] > h_tiled.shape[0]:
            y_tiled = y_tiled[: h_tiled.shape[0]]

        # Print shapes for debugging
        # print(f"Shape of y_tiled: {y_tiled.shape}")
        # print(f"Shape of h_tiled: {h_tiled.shape}")

        correlation = tf.reduce_mean(y_tiled * tf.math.conj(h_tiled))

        return correlation.numpy()

    def plot_cir_realization(self, a, tau):
        t = tau[0, 0, 0, :] / 1e-9  # Scale to ns
        a_abs = np.abs(a)[0, 0, 0, 0, 0, :, 0]
        a_max = np.max(a_abs)
        # Add dummy entry at start/end for nicer figure
        t = np.concatenate([(0.0,), t, (np.max(t) * 1.1,)])
        a_abs = np.concatenate([(np.nan,), a_abs, (np.nan,)])


def safe_numpy(tensor):
    try:
        return tensor.numpy()
    except:
        return tensor