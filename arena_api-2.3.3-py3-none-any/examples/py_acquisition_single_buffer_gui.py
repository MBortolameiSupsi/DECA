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

import numpy as np # pip3 install numpy
import cv2  # pip3 install opencv-python
from matplotlib import pyplot as plt # pip3 install matplotlib
# pip3 install pillow
from PIL import Image as PIL_Image
from PIL import ImageTk as PIL_ImageTk
# pip3 install tk / or 'sudo apt-get install python3-tk' for linux
from tkinter import *

from arena_api import enums
from arena_api.system import system
from arena_api.buffer import BufferFactory


'''
Acquisition: GUI
	This example introduces the basics of image acquisition. This
	includes setting image acquisition and buffer handling modes,
	setting the device to automatically negotiate packet size, and
	setting the stream packet resend node before starting the image
	stream. The example then acquires an image by grabbing and
	requeuing a single buffer and retrieving its data, before stopping
	the stream. It then displays the image using Matplotlib, OpenCV and Tkinter.
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


def convert_buffer_to_BGR8(buffer):
	'''
	Convert to BGR8 format, which allows images to be easily
	displayed
	'''
	if (buffer.pixel_format == enums.PixelFormat.BGR8):
		return buffer
	print('Converting image buffer pixel format to BGR8 ')
	return BufferFactory.convert(buffer, enums.PixelFormat.BGR8)
	


def show_image(buffer):
	"""
	Main factors to display images from arena_api:

	Pixel Format
	------------
	Buffer pixel arrangement should match the BGR8 arrangement.
	To get BGR8 arrangement in a buffer from the device:
	- set device to deliver BGR8 buffers before streaming by setting
			'PixelFormat' node to BGR8. This is a more efficient way because
			there is no need to convert the buffer afterward.
	- Or get any convertible pixel format from the device then convert it to
			BGR8 pixel formats using BufferFactory.convert().
	"""
	# Converting to BGR8 format
	print('\tConverting to BGR8 format')
	buffer_BGR8 = convert_buffer_to_BGR8(buffer)

	"""
	Multi Dimensional array
	-----------------------
	Pillow library can create Tkinter readable images using PIL.ImageTk
	class. However the data list must be a 3 dimensional array. Therefore,
	Numpy library is used to provides a way to reshape a 1D list to
	a 3D array.
	"""
	# Get a copy so it can be used after the buffer is requeued
	print('\tConvert image buffer to a numpy array')
	buffer_bytes_per_pixel = int(len(buffer_BGR8.data)/(buffer_BGR8.width * buffer_BGR8.height))
	np_array = np.asarray(buffer_BGR8.data, dtype=np.uint8)
	np_array_reshaped = np_array.reshape(buffer_BGR8.height, buffer_BGR8.width, buffer_bytes_per_pixel)

	
	print('Display image in the correct format for matplot')
	np_array_shaped_rgb = cv2.cvtColor(np_array_reshaped, cv2.COLOR_BGR2RGB)
	plt.imshow(np_array_shaped_rgb)
	plt.show()
	
	print('Display image using opencv')
	'''
	The array is already in BGR wich is what opencv is expecting
		so no conversion is needed like what is done for plt.imshow()
	'''
	cv2.imshow("window_title", np_array_reshaped)
	# Wait for user key before closing it
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	print('Display image using tk')
	# Creating \'PIL.Image\' instance from Numpy array')
	pil_image = PIL_Image.fromarray(np_array_reshaped)
	# 'Creating a Tkinter readable image from \'PIL.Image\' instance')
	root = Tk()
	pil_imagetk_photoimage = PIL_ImageTk.PhotoImage(pil_image)

	label = Label(root, image=pil_imagetk_photoimage)
	label.pack()
	root.mainloop()

	'''
	The buffer factory gives a converted copy of the device buffer, so
		destroy the image copy to prevent memory leaks
	'''
	BufferFactory.destroy(buffer_BGR8)


def example_entry_point():

	# Create a device
	devices = create_devices_with_tries()
	device = devices[0]
	print(f'Device used in the example:\n\t{device}')

	'''
	Store initial node values, return their values at the end of example
	'''
	nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])
	width_initial = nodes['Width'].value
	height_initial = nodes['Height'].value
	pixel_format_initial = nodes['PixelFormat'].value

	# Set width and Height to their max values
	print('Setting \'Width\' and \'Height\' Nodes value to their '
		'max values')
	nodes['Width'].value = nodes['Width'].max
	nodes['Height'].value = nodes['Height'].max

	tl_stream_nodemap = device.tl_stream_nodemap
	# Enable stream auto negotiate packet size
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	# Enable stream packet resend
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	'''
	Grab image -------------------------------------------------------------
	Starting the stream allocates buffers, which can be passed in as
		an argument (default: 10), and begins filling them with data.
	Buffers must later be requeued to avoid memory leaks.
	'''

	with device.start_stream():
		print(f'Stream started with 10 buffers')

		# 'Device.get_buffer()' with no arguments returns only one buffer
		print('\tGet one buffer')
		buffer = device.get_buffer()

		# Print some info about the image in the buffer
		print(f'\t\tbuffer received   | '
			f'Width = {buffer.width} pxl, '
			f'Height = {buffer.height} pxl, '
			f'Pixel Format = {buffer.pixel_format.name}')

		show_image(buffer)

		# Requeue to release buffer memory
		print('Requeuing device buffer')
		device.requeue_buffer(buffer)

	# Return nodes to initial values
	print("Return nodes to initial values")
	nodes['Width'].value = width_initial
	nodes['Height'].value = height_initial
	nodes['PixelFormat'].value = pixel_format_initial

	'''
	Clean up ----------------------------------------------------------------

	Stop stream and destroy device. This call is optional and will
		automatically be called for any remaining devices when the system
		module is unloading.
	'''
	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
