import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import matplotlib.pyplot as plt

INTEGER_BITS = 32
INTEGER_SCALE = 2**INTEGER_BITS;

def qam_modulation_manual(bits, f_carrier, samples_per_symbol, samplerate):
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

	return symbols

sample_rate = 500e+6	
phase_accumulator = 0.0
frequency_sweep_time = 50e-6
frequency_sweep_sample_rate = sample_rate
clocks_per_sample_time = int(frequency_sweep_time*frequency_sweep_sample_rate)
output_bits = 28        # this gives 1 Hz resolution
rom_depth = 2**output_bits
amplitude = 2**(output_bits - 1) - 1 # For signed output

default_frequency = 100e+6
sin_rom = amplitude * np.sin(np.linspace(0, 2 * np.pi, rom_depth, endpoint=False))
cos_rom = amplitude * np.cos(np.linspace(0, 2 * np.pi, rom_depth, endpoint=False))
delta_theta_default = (int(default_frequency)*2**output_bits)/frequency_sweep_sample_rate

accumulator_bits = output_bits*2
ftw = int(delta_theta_default*2**(accumulator_bits-output_bits))

dds_output_i = []
dds_output_q = []
ftw_storage = []

print('ftw = ', ftw)

for i in range(0, int(clocks_per_sample_time)):
	phase_accumulator = (phase_accumulator + ftw) % float(2**accumulator_bits)
#	print('PA = ', phase_accumulator, 'ftw = ', ftw, 'scale = ', 2**accumulator_bits)

	rom_index = int(phase_accumulator) >> (accumulator_bits - int(np.log2(rom_depth)))
	dds_output_i.append(sin_rom[rom_index])
	dds_output_q.append(cos_rom[rom_index])

rows = clocks_per_sample_time
cols = INTEGER_BITS
default_value = 0
i_binary = [[default_value for _ in range(cols)] for _ in range(rows)]
q_binary = [[default_value for _ in range(cols)] for _ in range(rows)]
i_binary_int = []
q_binary_int = []

for i in range(0, clocks_per_sample_time):
	i_binary_int.append(int(round(INTEGER_SCALE*dds_output_i[i], 0)))
	q_binary_int.append(int(round(INTEGER_SCALE*dds_output_q[i], 0)))

#for i in range(0, clocks_per_sample_time):
#	print("i = ", i_binary_int[i], "q = ", q_binary_int[i])

dds_time = np.arange(0, clocks_per_sample_time/frequency_sweep_sample_rate, 1/frequency_sweep_sample_rate)

plt.plot(dds_time, i_binary_int, color='blue')
plt.plot(dds_time, q_binary_int, color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Unfiltered Time Domain Signal")
plt.grid(True)
plt.savefig("scaled_sine_wave.jpg")
plt.show()

for i in range(0, clocks_per_sample_time):
	i_shift_register = i_binary_int[i]
	q_shift_register = q_binary_int[i]
	
	for j in range(0, INTEGER_BITS, 1):
		if (i_shift_register % 2 == 0):
			i_binary[i][j] = 0
		else:
			i_binary[i][j] = 1
			
		i_shift_register = i_shift_register >> 1
		
	for j in range(0, INTEGER_BITS, 1):
		if (q_shift_register % 2 == 0):
			q_binary[i][j] = 0
		else:
			q_binary[i][j] = 1
			
		q_shift_register = q_shift_register >> 1

# for i in range(0, clocks_per_sample_time):
# 	print("i = ", i, "i = ", end="")
# 	for j in range(0, INTEGER_BITS, 1):
# 		print(i_binary[i][j], end="")
# 	print("")
	
# 	print("i = ", i, "q = ", end="")
# 	for j in range(0, INTEGER_BITS, 1):
# 		print(q_binary[i][j], end="")
# 	print("")

binary_data_array = i_binary + q_binary

f_carrier = 125e+6
samples_per_symbol = 1
samplerate = 500e+6

modulated_data = [[default_value for _ in range(cols)] for _ in range(rows)]

for i in range(0, clocks_per_sample_time):
	modulated_data.append(qam_modulation_manual(binary_data_array[i][:], f_carrier, samples_per_symbol, samplerate))

