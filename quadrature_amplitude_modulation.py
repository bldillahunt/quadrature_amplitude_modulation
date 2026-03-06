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
UPSAMPLE_DATA_SIZE = SAMPLES_PER_SYMBOL*DATA_SIZE*(INTEGER_BITS/2)

def prbs_generator(seed_value):
	mask = 0xFFFFFFFF
	# Polynomial = x^32 + x^22 + x^2 + x^1 + 1
	lfsr_bit	= (seed_value ^ (seed_value >> 10) ^ (seed_value >> 30) ^ (seed_value >> 31)) & 0x1
	lfsr_data	= ((seed_value << 1) | (lfsr_bit)) & mask
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

def print_data_to_file(input_data, data_file):
	with open(data_file, 'w') as f:
		for item in input_data:
			f.write(f"{item}\n") # Write each item followed by a newline

def iq_time_domain_plot(signal_time, sample_rate, plot_title:str, i_input, q_input):
	signal_output_time = np.arange(0, signal_time, 1/sample_rate)
	plt.plot(signal_output_time, i_input, color='blue')
	plt.plot(signal_output_time, q_input, color='red')
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.title(plot_title)
	plt.grid(True)
	plt.show()

def time_domain_plot(signal_time, sample_rate, plot_title:str, signal_input):
	signal_output_time = np.arange(0, signal_time, 1/sample_rate)
	plt.plot(signal_output_time, signal_input, color='blue')
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.title(plot_title)
	plt.grid(True)
	plt.show()

# PRBS data generator
seed_value = 0xFFFFFFFF

random_data = []

# |---------- Transmit side ----------|

for i in range(0, DATA_SIZE):
	current_pattern = int(prbs_generator(seed_value))
	seed_value = current_pattern
	random_data.append(seed_value)
#	print(seed_value)

print('Data length = ', len(random_data))

prbs_data_file = 'prbs_data.txt'

with open(prbs_data_file, 'w') as f:
    for item in random_data:
        f.write(f"{hex(item)}\n") # Write each item followed by a newline

# Generate the local oscillator by implementing a DDS with I and Q outputs
tx_sample_rate = 2e+9	
tx_phase_accumulator = 0.0
tx_frequency_sweep_time = (DATA_SIZE*(INTEGER_BITS/2)*SAMPLES_PER_SYMBOL)/tx_sample_rate
tx_frequency_sweep_sample_rate = tx_sample_rate
tx_clocks_per_sample_time = DATA_SIZE*(INTEGER_BITS/2)*SAMPLES_PER_SYMBOL
tx_output_bits = 28        # this gives 1 Hz resolution
tx_rom_depth = 2**tx_output_bits
tx_amplitude = 2**(tx_output_bits - 1) - 1 # For signed output
tx_default_frequency = 500e+6

# Create sine and cosine ROMs that are 2^28 in size
tx_sin_rom = tx_amplitude * np.sin(np.linspace(0, 2 * np.pi, tx_rom_depth, endpoint=False))
tx_cos_rom = tx_amplitude * np.cos(np.linspace(0, 2 * np.pi, tx_rom_depth, endpoint=False))
tx_delta_theta_default = (int(tx_default_frequency)*2**tx_output_bits)/tx_frequency_sweep_sample_rate

print('Sin length = ', len(tx_sin_rom), 'Cos length = ', len(tx_cos_rom))

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

print('DDS out I = ', len(tx_dds_output_i), 'DDS out Q = ', len(tx_dds_output_q))

tx_dds_output = [complex(i, q) for i, q in zip(tx_dds_output_i, tx_dds_output_q)]

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

dds_data_file = 'dds_output_data.txt'

with open(dds_data_file, 'w') as f:
    for item in tx_dds_output:
        f.write(f"{item}\n") # Write each item followed by a newline

rows = int(tx_clocks_per_sample_time)
cols = INTEGER_BITS
default_value = 0
binary_data = [[default_value for _ in range(cols)] for _ in range(rows)]
symbols_array = []	#[[default_value for _ in range(int(cols/2))] for _ in range(rows)]
symbols_array_complex = []

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
	symbol_register = [complex(s) for s in symbols_array[i]]
	symbols_array_complex.append(symbol_register)

print('Finished symbols array')

symbols_array_file = 'symbols_array.txt'

with open(symbols_array_file, 'w') as f:
	for item in symbols_array_complex:
		f.write(f"{item}\n")

symbol_data_flat = []

# Flatten the two-dimensional array
for i in range(0, DATA_SIZE):
	for j in range(0, int(INTEGER_BITS/2)):
		symbol_data_flat.append(symbols_array_complex[i][j])

symbol_data_up = np.zeros(len(symbol_data_flat) * SAMPLES_PER_SYMBOL, dtype=complex)
symbol_data_up[::SAMPLES_PER_SYMBOL] = symbol_data_flat

print('Finished interpolation')

interpolated_data_file = 'interpolated_data.txt'

with open(interpolated_data_file, 'w') as f:
	for item in symbol_data_up:
		f.write(f"{item}\n")

x_coords = [c.real for c in symbol_data_up]
y_coords = [c.imag for c in symbol_data_up]

plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, color='red', marker='o')
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Argand Diagram of Complex Numbers")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

print('Finished interpolated signals plot')

nyquist_rate_new = 0.5 # Normalized to 1
cutoff_norm = nyquist_rate_new / SAMPLES_PER_SYMBOL

# Filter the data
# TX_N = 127         # Filter length (taps)
# tx_alpha = 0.35    # Roll-off factor
# tx_Ts = 500.0e-12  # Symbol duration
# tx_Fs = 1/tx_Ts		# Sampling rate (4 samples per symbol)
# fc = (1/SAMPLES_PER_SYMBOL)*tx_sample_rate

# # Generate filter coefficients and time vector
# t, tx_srrc_taps = rrcosfilter(TX_N, tx_alpha, tx_Ts, tx_Fs)

# n = np.arange(len(tx_srrc_taps))
# h_complex = tx_srrc_taps * np.exp(1j * 2 * np.pi * fc * n/tx_Ts)

# srrc_coefficient_file = 'srrc_coefficients.txt'

# with open(srrc_coefficient_file, 'w') as f:
# 	for item in h_complex:
# 		f.write(f"{item}\n")

# symbol_data_sum = symbol_data_up	#_i + symbol_data_up_q
# symbol_data_filtered = np.convolve(symbol_data_sum, h_complex, mode='full')

# Remove the group delay data
tx_reduced = []

#for i in range(TX_N-1, DATA_SIZE*int(INTEGER_BITS/2)*SAMPLES_PER_SYMBOL+TX_N-1):
#	tx_reduced.append(symbol_data_up[i])

x_coords_reduced = [c.real for c in symbol_data_up]
y_coords_reduced = [c.imag for c in symbol_data_up]

plt.figure(figsize=(6, 6))
plt.scatter(x_coords_reduced, y_coords_reduced, color='red', marker='o')
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Interpolated and reduced data")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

iq_time_domain_plot((DATA_SIZE*int(INTEGER_BITS/2)*SAMPLES_PER_SYMBOL)/tx_sample_rate, tx_sample_rate, 'Interpolated and reduced data', x_coords_reduced, y_coords_reduced)

reduced_filtered_file = 'reduced_filtered.txt'

with open(reduced_filtered_file, 'w') as f:
	for item in symbol_data_up:
		f.write(f"{item}\n")

# Mix the I and Q data with the DDS data
mixer_output = []

# Loop through the 65536 random complex values
for i in range(0, SAMPLES_PER_SYMBOL*DATA_SIZE*int(INTEGER_BITS/2)):
	# The original 32 bit data gets converted to 16 pairs
	mixer_output.append(symbol_data_up[i] * tx_dds_output[i])

transmitted_signal = mixer_output
print('Transmitted signal = ', len(transmitted_signal))

# Plot the I and Q data
x_coords_mixer = [c.real for c in mixer_output]
y_coords_mixer = [c.imag for c in mixer_output]

plt.figure(figsize=(6, 6))
plt.scatter(x_coords_mixer, y_coords_mixer, color='red', marker='o')
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Transmitted signal")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

x_coords_transmitter = [c.real for c in mixer_output]
y_coords_transmitter = [c.imag for c in mixer_output]

iq_time_domain_plot((DATA_SIZE*int(INTEGER_BITS/2)*SAMPLES_PER_SYMBOL)/tx_sample_rate, tx_sample_rate, 'Transmitted Signal', x_coords_transmitter, y_coords_transmitter)

# |---------- Receive side ----------|
rx_sample_rate = 2e+9	
rx_phase_accumulator = 0.0
rx_frequency_sweep_time = (DATA_SIZE*(INTEGER_BITS/2)*SAMPLES_PER_SYMBOL)/rx_sample_rate
rx_frequency_sweep_sample_rate = rx_sample_rate
rx_clocks_per_sample_time = int(rx_frequency_sweep_time*rx_frequency_sweep_sample_rate)
rx_output_bits = 28        # this gives 1 Hz resolution
rx_rom_depth = 2**rx_output_bits
rx_amplitude = 2**(rx_output_bits - 1) - 1 # For signed output
rx_default_frequency = 500e+6

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
	rx_rom_index = int(rx_phase_accumulator) >> (rx_accumulator_bits - int(np.log2(rx_rom_depth)))
	rx_dds_output_i.append(rx_sin_rom[rx_rom_index])
	rx_dds_output_q.append(rx_cos_rom[rx_rom_index])

rx_dds_output = [complex(i, q) for i, q in zip(rx_dds_output_i, rx_dds_output_q)]

rx_dds_output_time = np.arange(0, 2*rx_frequency_sweep_time, 1/rx_sample_rate)

plt.plot(rx_dds_output_time, rx_dds_output_i, color='blue')
plt.plot(rx_dds_output_time, rx_dds_output_q, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Receiver DDS output signal")
plt.grid(True)
plt.savefig("rx_dds_output.jpg")
plt.show()

tx_file_path = 'rx_dds_output.txt'

with open(tx_file_path, 'w') as f:
    for item in rx_dds_output:
        f.write(f"{item}\n") # Write each item followed by a newline

# Mix the RF signal with the local oscillator
rx_mixed = [x * y for x, y in zip(rx_dds_output, transmitted_signal)]

print_data_to_file(rx_mixed, 'rx_mixed.txt')
print('Mixed length', len(rx_mixed))

x_coords_mixed = [c.real for c in rx_mixed]
y_coords_mixed = [c.imag for c in rx_mixed]

plt.figure(figsize=(6, 6))
plt.scatter(x_coords_mixed, y_coords_mixed, color='red', marker='o')
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Receiver Mixer Output")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

# Filter the data
# RX_N = 127			# Filter length (taps)
# rx_alpha = 0.35		# Roll-off factor
# rx_Fs = 2e+9		# Sampling rate (4 samples per symbol)
# rx_Ts = 1/rx_Fs		# Symbol duration

# # Generate filter coefficients and time vector
# t, rx_srrc_taps = rrcosfilter(RX_N, rx_alpha, rx_Ts, rx_Fs)

# filtered_signal = np.convolve(rx_mixed, rx_srrc_taps, mode='full')

# rx_srrc_time = np.arange(0, (DATA_SIZE*SAMPLES_PER_SYMBOL*int(INTEGER_BITS/2))/(rx_sample_rate*SAMPLES_PER_SYMBOL), (1/(rx_sample_rate*SAMPLES_PER_SYMBOL)))

# # Remove the group delay from the data
# rx_reduced = []

# for i in range(RX_N-1, DATA_SIZE*SAMPLES_PER_SYMBOL*int(INTEGER_BITS/2)+RX_N-1):
# 	rx_reduced.append(filtered_signal[i])

rx_x_coords_reduced = [c.real for c in rx_mixed]
rx_y_coords_reduced = [c.imag for c in rx_mixed]

plt.figure(figsize=(6, 6))
plt.scatter(rx_x_coords_reduced, rx_y_coords_reduced, color='red', marker='o')
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Reduced Receiver Mixer Output")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

print('Reduced length', len(rx_mixed))

rx_data_sum = rx_mixed	# + rx_reduced_q

rx_data_sum_real = [c.real for c in rx_data_sum]
rx_data_sum_imag = [c.imag for c in rx_data_sum]

print_data_to_file(rx_data_sum, 'rx_data_sum.txt')
iq_time_domain_plot((DATA_SIZE*int(INTEGER_BITS/2)*SAMPLES_PER_SYMBOL)/rx_sample_rate, rx_sample_rate, 'Before anti-aliasing filter', rx_data_sum_real, rx_data_sum_imag)

print('RX data sum = ', len(rx_data_sum))

# Decimate to reduce to one sample per symbol
aaf = signal.decimate(rx_data_sum, SAMPLES_PER_SYMBOL, ftype='fir')
print('length of AAF = ', len(aaf))
print_data_to_file(aaf, 'anti_aliasing_filter.txt')
iq_time_domain_plot((DATA_SIZE*int(INTEGER_BITS/2))/rx_sample_rate, rx_sample_rate, 'After anti-aliasing filter', aaf.real, aaf.imag)
print('Finished anti-aliasing plot')

x_coords_aaf = [c.real for c in aaf]
y_coords_aaf = [c.imag for c in aaf]

plt.figure(figsize=(6, 6))
plt.scatter(x_coords_aaf, y_coords_aaf, color='red', marker='o')
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Anti-aliasing Filter Output")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

complex_waveform = aaf
magnitude_array = np.abs(complex_waveform)

phase_array = []

# Calculate the phase of each complex value
for i in range(0, len(aaf)):
	phase_array.append(math.atan2(aaf[i].real, aaf[i].imag))

rx_binary_data = []

# Loop through the complex waveform to obtain one of four locations
for i in range(0, len(complex_waveform)):
	if ((phase_array[i] >= 0) and (phase_array[i] < math.pi/2)):		# Quadrant I
		rx_binary_data.append(0x3)
	elif ((phase_array[i] >= math.pi/2) and (phase_array[i] < math.pi)):# Quadrant II
		rx_binary_data.append(0x1)
	elif (abs(phase_array[i]) > math.pi/2):								# Quadrant III
		rx_binary_data.append(0x0)
	elif (abs(phase_array[i]) <= math.pi/2):							# Quadrant IV
		rx_binary_data.append(0x2)

phase_array_file = 'phase_array.txt'

with open(phase_array_file, 'w') as f:
    for item in phase_array:
        f.write(f"{item}\n") # Write each item followed by a newline

binary_data_file = 'binary_data_array.txt'

with open(binary_data_file, 'w') as f:
    for item in rx_binary_data:
        f.write(f"{item}\n") # Write each item followed by a newline

decoded_data = []

# Put the 2 bit values together to form the 32 bit words of the original signal
for i in range(0, DATA_SIZE):
	decoder_register = 0;
	for j in range(0, int(INTEGER_BITS/2)):
		decoder_register = decoder_register | (((rx_binary_data[i*int(INTEGER_BITS/2)+j]) & 0x3) << INTEGER_BITS)
		decoder_register = decoder_register >> 2
#		print(hex(decoder_register))
#		char = sys.stdin.read(1)

	decoded_data.append(decoder_register)
#	print('i = ', i, 'data = ', decoder_register)

decoded_data_file = 'decoded_data.txt'

with open(decoded_data_file, 'w') as f:
	for item in decoded_data:
		f.write(f"{hex(item)}\n")

for i in range(0, DATA_SIZE):
	if (decoded_data[i] != random_data[i]):
		print('Error i = ', i, 'decoded data = ', decoded_data[i], 'PRBS data = ', random_data[i])
