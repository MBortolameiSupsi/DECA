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
from datetime import datetime

from arena_api.enums import PixelFormat
from arena_api.__future__.save import Writer
from arena_api.system import system
from arena_api.buffer import BufferFactory

'''
Save: Introduction
	This example introduces the basic save capabilities of the save library. It
	shows the construction of an image parameters object and an image writer, and
	saves a single image.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
pixel_format = PixelFormat.BGR8


def create_device_with_tries():
	'''
	Waits for the user to connect a device before raising
		an exception if it fails
	'''
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	devices = None
	while tries < tries_max:
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
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def save(buffer):
	'''
	demonstrates saving an image
	(1) converts image to a displayable pixel format
	(2) prepares image parameters
	(3) prepares image writer
	(4) saves image
	(5) destroys converted image
	'''
	'''
	Convert image
		Convert the image to a displayable pixel format. It is worth keeping in
		mind the best pixel and file formats for your application. This example
		converts the image so that it is displayable by the operating system.
	'''
	converted = BufferFactory.convert(buffer, pixel_format)
	print(f"{TAB1}Converted image to {pixel_format.name}")

	'''
	Prepare image writer
		The image writer requires 2 parameters to save an image: the buffer and
		specified file name or pattern. Default name for the image is
		'image_<count>.jpg' where count is a pre-defined tag that gets updated
		every time a buffer image.
	'''
	print(f'{TAB1}Prepare Image Writer')
	writer = Writer()
	writer.pattern = 'images/image_<count>.jpg'

	# Save converted buffer
	writer.save(converted)
	print(f'{TAB1}Image saved')

	# Destroy converted buffer to avoid memory leaks
	BufferFactory.destroy(converted)


def example_entry_point():
	devices = create_device_with_tries()
	device = devices[0]

	'''
	Setup stream values
	'''
	tl_stream_nodemap = device.tl_stream_nodemap
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	device.start_stream()

	buffer = device.get_buffer()

	save(buffer)

	device.requeue_buffer(buffer)

	# Clean up
	device.stop_stream()

	# Destroy Device
	system.destroy_device(device)


if __name__ == "__main__":
	print("Example Started\n")
	example_entry_point()
	print("\nExample Completed")
