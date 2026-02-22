import numpy as np
import matplotlib.pyplot as plt
import math
import sys
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

# Transmit side

for i in range(0, DATA_SIZE):
	current_pattern = np.int32(prbs_generator(seed_value))
	seed_value = current_pattern
	random_data.append(seed_value)
#	print('random data = ', random_data[i])

sample_rate = 500e+6	
phase_accumulator = 0.0
frequency_sweep_time = 131.072e-6
frequency_sweep_sample_rate = sample_rate
clocks_per_sample_time = int(frequency_sweep_time*frequency_sweep_sample_rate)
output_bits = 28        # this gives 1 Hz resolution
rom_depth = 2**output_bits
amplitude = 2**(output_bits - 1) - 1 # For signed output
default_frequency = 50e+6

# Create sine and cosine ROMs that are 2^28 in size
sin_rom = amplitude * np.sin(np.linspace(0, 2 * np.pi, rom_depth, endpoint=False))
cos_rom = amplitude * np.cos(np.linspace(0, 2 * np.pi, rom_depth, endpoint=False))
delta_theta_default = (int(default_frequency)*2**output_bits)/frequency_sweep_sample_rate

# Creat the oversized register that will hold extra bits to maintain high resolution
accumulator_bits = output_bits*2

# Shift the data up by 28 bits
ftw = int(delta_theta_default*2**(accumulator_bits-output_bits))

dds_output_i = []
dds_output_q = []
ftw_storage = []

print('ftw = ', ftw)

# Create the I and Q data arrays that have a size of 65536 each
for i in range(0, int(clocks_per_sample_time)):
	phase_accumulator = (phase_accumulator + ftw) % float(2**accumulator_bits)
#	print('PA = ', phase_accumulator, 'ftw = ', ftw, 'scale = ', 2**accumulator_bits)

	rom_index = int(phase_accumulator) >> (accumulator_bits - int(np.log2(rom_depth)))
	dds_output_i.append(sin_rom[rom_index])
	dds_output_q.append(cos_rom[rom_index])

print('Finished DDS data')

# Plot the I and Q data
dds_output_time = np.arange(0, frequency_sweep_time, 1/sample_rate)

plt.plot(dds_output_time, dds_output_i, color='blue')
plt.plot(dds_output_time, dds_output_q, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Unfiltered Time Domain Signal")
plt.grid(True)
plt.savefig("signal_time_domain.jpg")
plt.show()

rows = clocks_per_sample_time
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
		i_mixer_output.append(symbols_array[i][j].real * dds_output_i[i*32+j])
		q_mixer_output.append(symbols_array[i][j].imag * dds_output_q[i*32+j])

#		print('i = ', i, 'j = ', j, 'I = ', i_mixer_output[i*32+j].real, 'Q = ', q_mixer_output[i*32+j].imag, 'I DDS = ', dds_output_i[i*32+j], 'Q DDS = ', dds_output_q[i*32+j], 'symbols Real = ', symbols_array[i][j].real, 'symbols imag = ', symbols_array[i][j].imag)

transmit_time = np.arange(0, (DATA_SIZE*int(INTEGER_BITS/2))/sample_rate, 1/sample_rate)

plt.plot(transmit_time, i_mixer_output, color='blue')
plt.plot(transmit_time, q_mixer_output, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Unfiltered Time Domain Signal")
plt.grid(True)
plt.savefig("signal_time_domain.jpg")
plt.show()


# Receive side

