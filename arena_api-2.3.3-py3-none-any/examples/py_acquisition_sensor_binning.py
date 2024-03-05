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
import sys

from arena_api.system import system

'''
Acquisition: Sensor Binning
	This example demonstrates how to configure device settings to enable binning
	at the sensor level, so that the sensor will combine rectangles of pixels
	into larger "bins". This results in reduced resolution of images, but also
	reduces the amount of data sent to the software and networking layers.
'''


def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''

	tries = 0
	tries_max = 6
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


def configure_sensor_binning(device):
	'''
	Activate sensor binning and maximize bin size
	'''

	'''
	This is the entry we will use for BinningVerticalMode and
		BinningHorizontalMode. Sum will result in a brighter image, compared to
		Average.
	'''

	nodemap = device.nodemap
	BINTYPE = "Sum"

	binning_selector = nodemap.get_node("BinningSelector")
	binning_selector.value = "Sensor"

	binning_vertical_node = nodemap.get_node("BinningVertical")
	binning_horizontal_node = nodemap.get_node("BinningHorizontal")

	if (not binning_vertical_node.is_writable or
			not binning_horizontal_node.is_writable):
		print("Sensor binning is not supported: "
			"BinningVertical or BinningHorizontal not available")
		sys.exit()

	print("Find max bin height and width")
	bin_height = binning_vertical_node.max
	bin_width = binning_horizontal_node.max

	'''
	Set BinningHorizontal and BinningVertical to their maxes.
		This sets width and height of the bins: the number of pixels along each
		axis. Maximizes compression
	'''
	print(f"Set bin height and width to {bin_height} and {bin_width} respectively")
	binning_vertical_node.value = bin_height
	binning_horizontal_node.value = bin_width

	'''
	Sets binning mode
		Sets binning mode for horizontal and vertical axes. Generally, they're
		set to the same value.
	'''
	print(f"Set binning mode to {BINTYPE}")
	nodemap.get_node("BinningVerticalMode").value = BINTYPE
	nodemap.get_node("BinningHorizontalMode").value = BINTYPE


def get_multiple_image_buffers(device):

	number_of_buffers = 30

	# Starting the stream allocates buffers, which can be passed in as
	# an argument (default: 10), and begins filling them with data.
	# Buffers must later be requeued to avoid memory leaks.
	device.start_stream(number_of_buffers)
	print(f'Stream started with {number_of_buffers} buffers')

	# 'Device.get_buffer()' with no arguments returns one buffer(NOT IN A LIST)
	# 'Device.get_buffer(30)' returns 30 buffers(IN A LIST)
	print(f'\tGet {number_of_buffers} buffers in a list')
	buffers = device.get_buffer(number_of_buffers)

	# Print image buffer info
	for count, buffer in enumerate(buffers):
		print(f'\t\tbuffer{count:{2}} received | '
			f'Width = {buffer.width} pxl, '
			f'Height = {buffer.height} pxl, '
			f'Pixel Format = {buffer.pixel_format.name}')

	# 'Device.requeue_buffer()' takes a buffer or many buffers in a
	# list or tuple
	device.requeue_buffer(buffers)
	print(f'Requeued {number_of_buffers} buffers')

	device.stop_stream()
	print(f'Stream stopped')


def example_entry_point():

	# Get connected devices ---------------------------------------------------

	# Create a device
	devices = create_devices_with_tries()
	device = devices[0]
	print(f'Device used in the example:\n\t{device}')

	tl_stream_nodemap = device.tl_stream_nodemap

	# enable stream auto negotiate packet size
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

	# enable stream packet resend
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	'''
	Check if sensor binning is supported
	'''
	nodemap = device.nodemap
	binning_selector = nodemap.get_node("BinningSelector")
	print("Checking if sensor binning is supported")
	if("Sensor" not in binning_selector.enumentry_names or
			not binning_selector.enumentry_nodes.get("Sensor").is_readable):
		print("Sensor binning not supported by device: "
			"not available from BinningSelector")
		sys.exit()

	'''
	Store initial values
	'''
	acquisitionModeInitial = nodemap.get_node("AcquisitionMode").value
	binningSelectorInitial = nodemap.get_node("BinningSelector").value

	binningVerticalModeInitial = nodemap.get_node("BinningHorizontalMode").value
	binningHorizontalModeInitial = nodemap.get_node("BinningHorizontalMode").value

	binningVerticalInitial = nodemap.get_node("BinningVertical").value
	binningHorizontalInitial = nodemap.get_node("BinningHorizontal").value

	# Activate sensor binning

	configure_sensor_binning(device)

	# Grab images -------------------------------------------------------------

	get_multiple_image_buffers(device)

	# Clean up ----------------------------------------------------------------
	# Restore initial configurations
	print("Restoring initial configurations")
	nodemap.get_node("BinningVertical").value = binningVerticalInitial
	nodemap.get_node("BinningHorizontal").value = binningHorizontalInitial

	nodemap.get_node("BinningHorizontalMode").value = binningVerticalModeInitial
	nodemap.get_node("BinningHorizontalMode").value = binningHorizontalModeInitial

	nodemap.get_node("BinningSelector").value = binningSelectorInitial
	nodemap.get_node("AcquisitionMode").value = acquisitionModeInitial

	# Destroy device. Optional, implied by closing of module
	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
