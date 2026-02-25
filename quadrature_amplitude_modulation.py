import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import matplotlib.pyplot as plt
from scipy import signal
import commpy
from commpy.filters import rrcosfilter
import matplotlib.pyplot as plt

INTEGER_BITS = 32
INTEGER_SCALE = 2**INTEGER_BITS
DATA_SIZE = 2048
SAMPLES_PER_SYMBOL = 4

def prbs_generator(seed_value):
	# Polynomial = x^32 + x^22 + x^2 + x^1 + 1
	lfsr_bit	= (seed_value ^ (seed_value >> 10) ^ (seed_value >> 30) ^ (seed_value >> 31)) & 1
	lfsr_data	= (seed_value >> 1) | (lfsr_bit << 31)
#	print('bit = ', lfsr_bit, 'data = ', hex(lfsr_data))
#	char = sys.stdin.read(1)
	return lfsr_data

def qam_modulation_manual(bits):
	symbols = []
    
	for i in range(0, len(bits), 2):
		if bits[i] == 0 and bits[i+1] == 0:
			symbols.append(-1 - 1j)
		elif bits[i] == 0 and bits[i+1] == 1:
			symbols.append(-1 + 1j)
		elif bits[i] == 1 and bits[i+1] == 0:
			symbols.append(1 - 1j)
		else:
			symbols.append(1 + 1j)

#	print('symbols = ', symbols)
	return symbols

# PRBS data generator
seed_value = 0xFFFFFFFF

random_data = []

# |---------- Transmit side ----------|

for i in range(0, DATA_SIZE):
	current_pattern = np.int32(prbs_generator(seed_value))
	seed_value = current_pattern
	random_data.append(seed_value)
#	print('random data = ', random_data[i])

tx_sample_rate = 500e+6	
tx_phase_accumulator = 0.0
tx_frequency_sweep_time = 131.072e-6
tx_frequency_sweep_sample_rate = tx_sample_rate
tx_clocks_per_sample_time = int(tx_frequency_sweep_time*tx_frequency_sweep_sample_rate)
tx_output_bits = 28        # this gives 1 Hz resolution
tx_rom_depth = 2**tx_output_bits
tx_amplitude = 2**(tx_output_bits - 1) - 1 # For signed output
tx_default_frequency = 50e+6

# Create sine and cosine ROMs that are 2^28 in size
tx_sin_rom = tx_amplitude * np.sin(np.linspace(0, 2 * np.pi, tx_rom_depth, endpoint=False))
tx_cos_rom = tx_amplitude * np.cos(np.linspace(0, 2 * np.pi, tx_rom_depth, endpoint=False))
tx_delta_theta_default = (int(tx_default_frequency)*2**tx_output_bits)/tx_frequency_sweep_sample_rate

# Creat the oversized register that will hold extra bits to maintain high resolution
tx_accumulator_bits = tx_output_bits*2

# Shift the data up by 28 bits
tx_ftw = int(tx_delta_theta_default*2**(tx_accumulator_bits-tx_output_bits))

tx_dds_output_i = []
tx_dds_output_q = []
tx_ftw_storage = []

print('ftw = ', tx_ftw)

# Create the I and Q data arrays that have a size of 65536 each
for i in range(0, int(tx_clocks_per_sample_time)):
	tx_phase_accumulator = (tx_phase_accumulator + tx_ftw) % float(2**tx_accumulator_bits)
#	print('PA = ', phase_accumulator, 'ftw = ', ftw, 'scale = ', 2**accumulator_bits)

	tx_rom_index = int(tx_phase_accumulator) >> (tx_accumulator_bits - int(np.log2(tx_rom_depth)))
	tx_dds_output_i.append(tx_sin_rom[tx_rom_index])
	tx_dds_output_q.append(tx_cos_rom[tx_rom_index])

print('Finished DDS data')

# Plot the I and Q data
dds_output_time = np.arange(0, tx_frequency_sweep_time, 1/tx_sample_rate)

plt.plot(dds_output_time, tx_dds_output_i, color='blue')
plt.plot(dds_output_time, tx_dds_output_q, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("DDS output signal")
plt.grid(True)
plt.savefig("signal_time_domain.jpg")
plt.show()

rows = tx_clocks_per_sample_time
cols = INTEGER_BITS
default_value = 0
binary_data = [[default_value for _ in range(cols)] for _ in range(rows)]
symbols_array = []	#[[default_value for _ in range(int(cols/2))] for _ in range(rows)]

# Loop through the random data (size = 2048)
for i in range(0, DATA_SIZE):
	shift_register = random_data[i]

	# Convert the random data to binary (32 bits)
	for j in range(0, INTEGER_BITS):
		binary_data[i][j] = shift_register & 0x00000001
		shift_register = shift_register >> 1
	
	# Convert the 32 bit value to complex numbers (1+1*j, -1+1*j, -1-1*j, 1-1*j)
	# This reduces the array to 16 complex numbers
	symbols_array.append(qam_modulation_manual(binary_data[i]))

print('Finished symbols array')

# Mix the I and Q data with the DDS data
i_mixer_output = []
q_mixer_output = []

# Loop through the 65536 random complex values
for i in range(0, DATA_SIZE):
	# The original 32 bit data gets converted to 16 pairs
	for j in range(0, int(INTEGER_BITS/2)):
		i_mixer_output.append(symbols_array[i][j].real * tx_dds_output_i[i*32+j])
		q_mixer_output.append(symbols_array[i][j].imag * tx_dds_output_q[i*32+j])

#		print('i = ', i, 'j = ', j, 'I = ', i_mixer_output[i*32+j].real, 'Q = ', q_mixer_output[i*32+j].imag, 'I DDS = ', dds_output_i[i*32+j], 'Q DDS = ', dds_output_q[i*32+j], 'symbols Real = ', symbols_array[i][j].real, 'symbols imag = ', symbols_array[i][j].imag)

# Upsample the data to get 4 samples per symbol
i_data_up = np.zeros(len(i_mixer_output) * SAMPLES_PER_SYMBOL)
q_data_up = np.zeros(len(q_mixer_output) * SAMPLES_PER_SYMBOL)
i_data_up[::SAMPLES_PER_SYMBOL] = i_mixer_output
q_data_up[::SAMPLES_PER_SYMBOL] = q_mixer_output

# 2. Design an interpolation low-pass FIR filter
# The cutoff frequency should be 1/SAMPLES_PER_SYMBOL of the original Nyquist frequency
# We use firwin to design a linear-phase filter
# The filter cutoff is specified as a normalized frequency (0 to 1, where 1 is Nyquist)
# For interpolation, the cutoff should be 1/SAMPLES_PER_SYMBOL relative to the *new* Nyquist frequency
# (which is the original Nyquist * SAMPLES_PER_SYMBOL), so we use 1/SAMPLES_PER_SYMBOL as the normalized cutoff.
nyquist_rate_new = 0.5 # Normalized to 1
cutoff_norm = nyquist_rate_new / SAMPLES_PER_SYMBOL

# Design filter coefficients, scale the taps by L to maintain signal amplitude
#taps = signal.firwin(127, cutoff_norm, window='hamming', scale=SAMPLES_PER_SYMBOL)

# 3. Apply the FIR filter to the upsampled signal
# lfilter performs a discrete convolution with the taps
#i_data_filtered = signal.lfilter(taps, 1.0, i_data_up)
#q_data_filtered = signal.lfilter(taps, 1.0, q_data_up)

# Filter the data
TX_N = 127         # Filter length (taps)
tx_alpha = 0.35    # Roll-off factor
tx_Ts = 500.0e-12  # Symbol duration
tx_Fs = 1/tx_Ts		# Sampling rate (8 samples per symbol)

# Generate filter coefficients and time vector
t, tx_srrc_taps = rrcosfilter(TX_N, tx_alpha, tx_Ts, tx_Fs)

i_data_filtered = np.convolve(i_data_up, tx_srrc_taps, mode='full')
q_data_filtered = np.convolve(q_data_up, tx_srrc_taps, mode='full')

# Remove the group delay data
tx_reduced_i = []
tx_reduced_q = []

for i in range(TX_N-1, DATA_SIZE*SAMPLES_PER_SYMBOL*int(INTEGER_BITS/2)+TX_N-1):
	tx_reduced_i.append(i_data_filtered[i])
	tx_reduced_q.append(q_data_filtered[i])

transmit_time = np.arange(0, (DATA_SIZE*SAMPLES_PER_SYMBOL*int(INTEGER_BITS/2))/(tx_sample_rate*SAMPLES_PER_SYMBOL), (1/(tx_sample_rate*SAMPLES_PER_SYMBOL)))

plt.plot(transmit_time, tx_reduced_i, color='blue')
plt.plot(transmit_time, tx_reduced_q, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Complex output signal")
plt.grid(True)
plt.savefig("TX filter output.jpg")
plt.show()

transmitted_signal = tx_reduced_i + tx_reduced_q

# |---------- Receive side ----------|
rx_sample_rate = 500e+6	
rx_phase_accumulator = 0.0
rx_frequency_sweep_time = 131.072e-6
rx_frequency_sweep_sample_rate = rx_sample_rate
rx_clocks_per_sample_time = int(rx_frequency_sweep_time*rx_frequency_sweep_sample_rate)
rx_output_bits = 28        # this gives 1 Hz resolution
rx_rom_depth = 2**rx_output_bits
rx_amplitude = 2**(rx_output_bits - 1) - 1 # For signed output
rx_default_frequency = 50e+6

# Create sine and cosine ROMs that are 2^28 in size
rx_sin_rom = rx_amplitude * np.sin(np.linspace(0, 2 * np.pi, rx_rom_depth, endpoint=False))
rx_cos_rom = rx_amplitude * np.cos(np.linspace(0, 2 * np.pi, rx_rom_depth, endpoint=False))
rx_delta_theta_default = (int(rx_default_frequency)*2**rx_output_bits)/rx_frequency_sweep_sample_rate

# Create the oversized register that will hold extra bits to maintain high resolution
rx_accumulator_bits = rx_output_bits*2

# Shift the data up by 28 bits
rx_ftw = int(rx_delta_theta_default*2**(rx_accumulator_bits-rx_output_bits))

rx_dds_output_i = []
rx_dds_output_q = []
rx_ftw_storage = []

print('ftw = ', rx_ftw)

# Create the I and Q data arrays that have a size of 65536 each
for i in range(0, int(rx_clocks_per_sample_time*2)):
	rx_phase_accumulator = (rx_phase_accumulator + rx_ftw) % float(2**rx_accumulator_bits)
#	print('PA = ', phase_accumulator, 'ftw = ', ftw, 'scale = ', 2**accumulator_bits)

	rx_rom_index = int(rx_phase_accumulator) >> (rx_accumulator_bits - int(np.log2(rx_rom_depth)))
	rx_dds_output_i.append(rx_sin_rom[rx_rom_index])
	rx_dds_output_q.append(rx_cos_rom[rx_rom_index])
#	print('PA = ', rx_phase_accumulator, 'index = ', rx_rom_index, 'I = ', rx_sin_rom[rx_rom_index], 'Q = ', (rx_cos_rom[rx_rom_index]))

tx_file_path_i = 'dds_output_i.txt'

with open(tx_file_path_i, 'w') as f:
    for item in rx_dds_output_i:
        f.write(f"{item}\n") # Write each item followed by a newline

tx_file_path_q = 'dds_output_q.txt'

with open(tx_file_path_q, 'w') as f:
    for item in rx_dds_output_q:
        f.write(f"{item}\n") # Write each item followed by a newline

# Mix the RF signal with the local oscillator
mixed_i = []	# rx_dds_output_i * transmitted_signal
mixed_q = []	# rx_dds_output_q * transmitted_signal

mixed_i = [x * y for x, y in zip(rx_dds_output_i, transmitted_signal)]
mixed_q = [x * y for x, y in zip(rx_dds_output_q, transmitted_signal)]

transmit_time = np.arange(0, (DATA_SIZE*SAMPLES_PER_SYMBOL*int(INTEGER_BITS/2))/(rx_sample_rate*SAMPLES_PER_SYMBOL), (1/(rx_sample_rate*SAMPLES_PER_SYMBOL)))

plt.plot(transmit_time, mixed_i, color='blue')
plt.plot(transmit_time, mixed_q, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Down converted signal")
plt.grid(True)
plt.savefig("down_converted.jpg")
plt.show()

# Filter the data
RX_N = 127         # Filter length (taps)
rx_alpha = 0.35    # Roll-off factor
rx_Ts = 500.0e-12  # Symbol duration
rx_Fs = 1/rx_Ts		# Sampling rate (8 samples per symbol)

# Generate filter coefficients and time vector
t, rx_srrc_taps = rrcosfilter(RX_N, rx_alpha, rx_Ts, rx_Fs)

filtered_signal_i = np.convolve(mixed_i, rx_srrc_taps, mode='full')
filtered_signal_q = np.convolve(mixed_q, rx_srrc_taps, mode='full')

rx_srrc_time = np.arange(0, (DATA_SIZE*SAMPLES_PER_SYMBOL*int(INTEGER_BITS/2))/(rx_sample_rate*SAMPLES_PER_SYMBOL), (1/(rx_sample_rate*SAMPLES_PER_SYMBOL)))

# Remove the group delay from the data
rx_reduced_i = []
rx_reduced_q = []

for i in range(RX_N-1, DATA_SIZE*SAMPLES_PER_SYMBOL*int(INTEGER_BITS/2)+RX_N-1):
	rx_reduced_i.append(filtered_signal_i[i])
	rx_reduced_q.append(filtered_signal_q[i])

plt.plot(rx_srrc_time, rx_reduced_i, color='blue')
plt.plot(rx_srrc_time, rx_reduced_q, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Output of SRRC")
plt.grid(True)
plt.savefig("rx_srrc_output.jpg")
plt.show()



