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

from arena_api.__future__.save import Writer

'''
Lookup Tables: Introduction
	This example introduces the lookup tables (LUT), which are used to transform
	image data into a desired output format. LUTs give an output value for each
	of a range of index values. This example enables a lookup table node to
	invert the intensity of a single image. This is done by accessing the LUT
	index node and setting the LUT node values to the newly calculated pixel
	intensity value. It takes some time to update each pixel with the new value.
	The example then saves the new image by saving to the image writer.
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


def save_one_image(device, filename):
	with device.start_stream():
		writer = Writer()
		print(f'Stream started with 10 buffers')

		# 'Device.get_buffer()' with no arguments returns only one buffer
		print('\tGet one buffer')
		buffer = device.get_buffer()

		# Print some info about the image in the buffer
		print(f'\t\tbuffer received   | '
			f'Width = {buffer.width} pxl, '
			f'Height = {buffer.height} pxl, '
			f'Pixel Format = {buffer.pixel_format.name}')

		print("\t\tSave buffer")
		writer.save(buffer, filename)

		# Requeue the image buffer
		device.requeue_buffer(buffer)
		print("\tBuffer requeued")


def example_entry_point():

	# Settings
	SLOPE = -1

	# Create a device
	devices = create_devices_with_tries()
	device = devices[0]
	print(f'Device used in the example:\n\t{device}')

	nodemap = device.nodemap

	# Store initial settings, to restore later
	initial_lut_enable = nodemap.get_node("LUTEnable").value
	initial_acquisition_mode = nodemap.get_node("AcquisitionMode").value

	nodemap.get_node("LUTEnable").value = False

	'''
	Save one image with LUT disabled
	'''
	save_one_image(device, "Images/lut_no.png")

	'''
	Enable and configure LUT
	'''
	node_lut_index = nodemap.get_node("LUTIndex")
	node_lut_value = nodemap.get_node("LUTValue")

	if (node_lut_index is None or node_lut_value is None):
		raise Exception("Requisite node(s) LUTIndex and/or LUTValue do(es) not exist")

	nodemap.get_node("LUTEnable").value = True

	for i in range(node_lut_index.max):
		'''
		Select each pixel's intesity, and map it to its inversion
			i -> max - i, using example's original settings (SLOPE = -1)
		'''
		node_lut_index.value = i
		node_lut_value.value = SLOPE * i + node_lut_index.max

		if (i % 1024 == 0):
			print("\t", end="")

		if (i % 256 == 255):
			print(".", end="")

		if (i % 1024 == 1023):
			print()

	'''
	Save image with LUT enabled: with inverted intensity
	'''
	print()
	save_one_image(device, "Images/lut_yes.png")

	nodemap.get_node("LUTEnable").value = initial_lut_enable
	nodemap.get_node("AcquisitionMode").value = initial_acquisition_mode

	# Destroy devices. Redundant: implicit destroy occurs when module closes
	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
