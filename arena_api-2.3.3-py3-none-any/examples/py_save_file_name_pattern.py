# -----------------------------------------------------------------------------
# Copyright (c) 2022, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED AS IS WITHOUT WARRANTY OF ANY KIND,
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
Save: File Name Pattern
	This example demonstrates saving a set of images according to a file name
	pattern, which uses the <count> and <timestamp> tags to differentiate between
	saved images. The essential points of the example include setting the image
	writer up with a file name pattern and using the cascading I/O operator (<<)
	to update the timestamp and save each image.
'''

'''
Generator functions
'''


def get_vendor(device):
	'''
	Generator function for vendor
	'''
	while True:
		yield device.nodemap.get_node("DeviceVendorName").value


def get_model(device):
	'''
	Generator function for model name
	'''
	while True:
		yield device.nodemap.get_node("DeviceModelName").value


def get_serial(device):
	'''
	Generator function for serial number
	'''
	while True:
		yield device.nodemap.get_node("DeviceSerialNumber").value


def get_and_save_images(device, writer, num_images):

	# Starting the stream allocates buffers, which can be passed in as
	# an argument (default: 10), and begins filling them with data.
	# Buffers must later be requeued to avoid memory leaks.
	with device.start_stream(num_images):
		print(f'Stream started with {num_images} buffers')
		for i in range(num_images):
			# 'Device.get_buffer()' with no arguments returns only one buffer
			print('\tGet one buffer')
			buffer = device.get_buffer()

			# Print some info about the image in the buffer
			print(f'\t\tbuffer received   | '
				f'Width = {buffer.width} pxl, '
				f'Height = {buffer.height} pxl, '
				f'Pixel Format = {buffer.pixel_format.name}')

			print(f"\t\tSave image {i}")
			writer.save(buffer)

			# Requeue the image buffer
			device.requeue_buffer(buffer)


def create_devices_with_tries():
	# Connect a device
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
					'.' * sec_count, end='\\r')
			tries += 1
		else:
			print(f'Created {len(devices)} device(s)\\n')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def example_entry_point():
	# Settings

	'''
	File name pattern
		File name patterns can use tags to easily customize your file names.
		Customizable tags can be added to a file name pattern and later set on
		the fly. Two tags, <count> and <datetime> have been built in to the save
		library. As seen below, <datetime> can take an argument to specify
		output. <count> also accepts arguments (local, path, and global) to
		specify what exactly is being counted.
	'''
	FILE_NAME_PATTERN = "Images/py_save_file_name_pattern/<vendor>_<model>_<serial>\
_image<count>-<datetime:yyMMdd_hhmmss_fff>.bmp"

	# number of images to acquire and save
	NUM_IMAGES = 25

	# image timeout (milliseconds)
	TIMEOUT = 2000

	devices = create_devices_with_tries()
	device = devices[0]

	writer = Writer()

	'''
	Must register tags with writer before including them in pattern
		Must include a generator function
	'''
	print("Register tags")
	writer.register_tag("vendor", generator=get_vendor(device))
	writer.register_tag("model", generator=get_model(device))
	writer.register_tag("serial", generator=get_serial(device))

	print("Set file name pattern")
	writer.pattern = FILE_NAME_PATTERN

	get_and_save_images(device, writer, NUM_IMAGES)
	system.destroy_device()


if __name__ == '__main__':
	print('WARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('Example started')
	example_entry_point()
	print('Example finished successfully')
