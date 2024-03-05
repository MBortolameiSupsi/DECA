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
Chunk Data: Introduction
	This example introduces the configuration and retrieval of chunk data. Chunk
	data is data that has been tacked on to the end of image data in order to
	provide useful information on the image. Configuring chunk data involves
	activating chunk mode and enabling desired chunks. Retrieving chunk data from
	an image is similar to retrieving nodes from a node map.
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


def configure_and_retrieve_chunk_data(device):
	'''
		Demonstrates chunk data
		(1) Activates chunk mode
		(2) Enable selected chunks
		(3) Starts the stream and gets images
		(4) Retrieves chunk data from image
		(5) Requeues buffers and stops the stream
	'''

	# Activating chunk data ---------------------------------------------------
	nodemap = device.nodemap

	# Store initial ChunkModeActive to restore later
	initial_chunk_mode_active = nodemap.get_node("ChunkModeActive").value

	# Prep device to send chunk data with the buffer
	print('Activating chunk data on device')
	nodemap.get_node('ChunkModeActive').value = True

	'''
	List of chunk names to send with the buffer.
		Use 'ChunkSelector' node to select the chunk to send. This is done by
		setting the value of 'ChunkSelector' node to the name of the chunk, then
		enabling the chunk by setting 'ChunkEnable' to 'True'
	'''
	chunks_to_append_to_buffer = ['PixelFormat', 'CRC',
								'Width', 'Height',
								'OffsetX', 'OffsetY']

	chunk_selector_node = nodemap.get_node('ChunkSelector')
	chunk_enable_node = nodemap.get_node('ChunkEnable')

	# Used to store initial values
	initial_chunk_selector = chunk_selector_node.value
	initial_chunk_enable = {}

	for chunk_name in chunks_to_append_to_buffer:
		print(f'\tsetting \'ChunkSelector\' node value to \'{chunk_name}\'')
		chunk_selector_node.value = chunk_name
		initial_chunk_enable[chunk_name] = chunk_enable_node.value
		print(f'\tenabling \'{chunk_name}\' by setting \'ChunkEnable\' '
			f'node value to \'True\'')
		chunk_enable_node.value = True

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
		To access the chunks that are appended to a buffer use buffer
			'Buffer.get_chunk()' method. The chunk name to get is the same as the
			chunk value that was set and enabled but preceded with 'Chunk'.
		'''
		chunk_node_names = ['ChunkPixelFormat', 'ChunkCRC',
							'ChunkWidth', 'ChunkHeight',
							'ChunkOffsetX', 'ChunkOffsetY']

		# Iterate over every chunk data buffer and print chunks' value
		for buffer_index, chunk_data_buffer in enumerate(chunk_data_buffers):
			'''
			Get all chunks in list from buffer
				'Buffer.get_chunk()' may raise an exception because the buffer is
				incomplete
			'''
			if chunk_data_buffer.is_incomplete:
				print(f'\t\t---------------------------------------------')
				print(f'\t\tChunk data buffer{buffer_index} is incomplete')
				print(f'\t\t---------------------------------------------')
				# Continue
			try:
				chunk_nodes_dict = chunk_data_buffer.get_chunk(
					chunk_node_names)

				# Print the value of the chunks for the current buffer
				print(f'\t\tChunk data buffer{buffer_index} chunks value:')
				for chunk_node_name, chunk_node in chunk_nodes_dict.items():
					print(f'\t\t\t{chunk_node_name} = {chunk_node.value}')
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
	for chunk_name in chunks_to_append_to_buffer:
		chunk_selector_node.value = chunk_name
		chunk_enable_node.value = initial_chunk_enable[chunk_name]

	chunk_selector_node.value = initial_chunk_selector
	nodemap.get_node("ChunkModeActive").value = initial_chunk_mode_active


def example_entry_point():

	# Create a device
	devices = create_devices_with_tries()
	device = devices[0]
	print(f'Device used in the example:\n\t{device}')

	tl_stream_nodemap = device.tl_stream_nodemap

	# Enable stream auto negotiate packet size
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

	# Enable stream packet resend
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	# Run example
	configure_and_retrieve_chunk_data(device)

	# Clean up ----------------------------------------------------------------

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
