# -----------------------------------------------------------------------------
# Copyright (c) 2022, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

import time

from arena_api.system import system

'''
Chunk Data: CRC Validation
	This example demonstrates the use of chunk data to verify data through a
	Cyclical Redundancy Check (or CRC for short). CRCs are meant to check the
	validity of sent data. It is performed by doing a series of calculations on
	the raw data before and after it is sent. If the resultant integer values
	match, then it is safe to assume the integrity of the data.
'''


def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''

	tries = 0
	tries_max = 1
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			print(
				f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				print(f'{sec_count + 1 } seconds passed ',
					'.' * sec_count, end='\r')
			tries += 1
		else:
			print(f'Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def configure_retrieve_and_validate_chunk_data(device):
	'''
	Demonstrates chunk data
		(1) Activates chunk mode
		(2) Enable selected chunks
		(3) Starts the stream and gets images
		(4) Retrieves exposure and gain chunk data from image
		(5) Requeues buffers and stops the stream
	'''

	# Activating chunk data ---------------------------------------------------
	nodemap = device.nodemap

	# Store initial ChunkModeActive to restore later
	initial_chunk_mode_active = nodemap.get_node("ChunkModeActive").value

	'''
	Prep device to send chunk data with the buffer
	'''
	initial_chunk_mode_active = nodemap.get_node("ChunkModeActive").value
	print('Activating chunk data on device')
	nodemap.get_node('ChunkModeActive').value = True

	# Use CRC chunk: unnecessary, any chunk works
	initial_chunk_selector = nodemap.get_node('ChunkSelector').value
	print(f"\tSetting 'ChunkSelector' node value to 'CRC'")
	nodemap.get_node("ChunkSelector").value = "CRC"

	initial_chunk_enable = nodemap.get_node('ChunkEnable').value
	print(f"\tEnabling CRC by setting 'ChunkEnable' node value to 'True'")
	nodemap.get_node("ChunkEnable").value = True

	# Grab chunk data buffers -------------------------------------------------

	# Starting the stream allocates buffers and begins filling them with data.
	with device.start_stream():
		print(f'Stream started with 10 buffers')
		'''
		Buffers contains chunk data that was enabled.
			'Device.get_buffer()' with no arguments returns only one buffer
			'Device.get_buffer()' with a number greater than 1 returns a list of
			buffers
		'''
		print('\tGet chunk data buffer(s)')

		# This would timeout or returns all of the 10 buffers
		chunk_data_buffers = device.get_buffer(number_of_buffers=10)
		print(f'\t{len(chunk_data_buffers)} chunk data buffers received')

		'''
		Confirm that each chunk is complete, then validate the buffer's CRC
		'''
		for buffer_index, chunk_data_buffer in enumerate(chunk_data_buffers):
			if chunk_data_buffer.is_incomplete:
				print(f'\t\t---------------------------------------------')
				print(f'\t\tChunk data buffer{buffer_index} is incomplete')
				print(f'\t\t---------------------------------------------')
				# Continue
			try:
				'''
				Buffer.is_valid_crc: compares calculated CRC value to the chunk CRC value.
					This only works with buffers retrieved in chunk mode.
				'''
				print(f"\t\tChunk data CRC is valid: {chunk_data_buffer.is_valid_crc}")
			except ValueError:
				print(f'\t\t\tFailed to get chunks')
				print(f'\t\t---------------------------------------------')
				continue

		# Requeue the chunk data buffers
		device.requeue_buffer(chunk_data_buffers)

	'''
	When the scope of the context manager ends, then 'Device.stop_stream()'
		is called automatically
	'''
	print('Stream stopped')

	# Restore initial values
	nodemap.get_node("ChunkEnable").value = initial_chunk_enable
	nodemap.get_node("ChunkSelector").value = initial_chunk_selector
	nodemap.get_node("ChunkModeActive").value = initial_chunk_mode_active


def example_entry_point():

	# Create a device
	devices = create_devices_with_tries()
	device = devices[0]
	print(f'Device used in the example:\n\t{device}')
	nodemap = device.nodemap
	initial_acquisition_mode = nodemap.get_node("AcquisitionMode").value
	nodemap.get_node("AcquisitionMode").value = "Continuous"

	tl_stream_nodemap = device.tl_stream_nodemap

	# Enable stream auto negotiate packet size
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

	# Enable stream packet resend
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	# Run example
	configure_retrieve_and_validate_chunk_data(device)

	# Clean up ----------------------------------------------------------------

	nodemap.get_node("AcquisitionMode").value = initial_acquisition_mode
	'''
	Destroy all created devices. This call is optional and will
		automatically be called for any remaining devices when the system module
		is unloading.
	'''
	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
