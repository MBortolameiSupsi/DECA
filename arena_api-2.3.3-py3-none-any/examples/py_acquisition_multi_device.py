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

import threading
import time

from arena_api.system import system

# Number of buffers allocated for a device stream
NUMBER_OF_BUFFERS = 25

'''
Acquisition: Multi-Device
	This example introduces the basics of image acquisition for multiple devices
	and creating multiple threads. This includes creating all discovered devices,
	creating a thread for each device to acquire images. The thread then starts
	acquiring images by grabbing and requeuing buffers, before finally stopping
	the image stream.
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
			print(f'Created {len(devices)} device(s)\n')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def safe_print(*args, **kwargs):
	'''
	This function ensures resource access is locked to a single thread
	'''
	with threading.Lock():
		print(*args, **kwargs)


def get_multiple_image_buffers(device):
	'''
	This function demonstrates an acquisition on a device

	(1) Start stream with N buffers
	(2) Print each buffer info
	(3) Requeue each buffer
	'''

	thread_id = f'''{device.nodemap['DeviceModelName'].value}''' \
		f'''-{device.nodemap['DeviceSerialNumber'].value} |'''

	# Start stream with 25 buffers
	device.start_stream(NUMBER_OF_BUFFERS)

	# Print image buffer info
	for count in range(NUMBER_OF_BUFFERS):
		buffer = device.get_buffer()

		safe_print(f'{thread_id:>30}',
				f'\tbuffer{count:{2}} received | '
				f'Width = {buffer.width} pxl, '
				f'Height = {buffer.height} pxl, '
				f'Pixel Format = {buffer.pixel_format.name}')

		
		'''
		`Device.requeue_buffer()` takes a buffer or many buffers in a list or
		tuple
		'''
		device.requeue_buffer(buffer)
		safe_print(f'{thread_id:>30}', f'\tbuffer{count:{2}} requeued | ')

	device.stop_stream()


def example_entry_point():
	'''
	This function introduces the basics of image acquisition for multiple
		devices and creating multiple threads. This includes creating all
		discovered devices, creating a thread for each device to acquire images.
		The thread then starts acquiring images by grabbing and requeuing
		buffers, before finally stopping the image stream.
	'''
	# Create devices
	devices = create_devices_with_tries()

	# Create empty lists for the devices
	thread_list = []
	initial_acquisition_mode_list = []

	# Configure each device
	for device in devices:
		# Get nodemap
		nodemap = device.nodemap

		# Store initial acquisition mode to restore later
		initial_acquisition_mode_list.append(nodemap.get_node("AcquisitionMode").value)

		# Set acquisition mode to continuous
		nodemap.get_node("AcquisitionMode").value = "Continuous"

		# Get device stream nodemap
		tl_stream_nodemap = device.tl_stream_nodemap

		# Set buffer handling mode to "Newest First"
		tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"

		# Enable stream auto negotiate packet size
		tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

		# Enable stream packet resend
		tl_stream_nodemap['StreamPacketResendEnable'].value = True

	# Create a thread for each device
	for device in devices:
		thread = threading.Thread(target=get_multiple_image_buffers,
								args=(device,))
		# Add a thread to the thread list
		thread_list.append(thread)

	# Start each thread in the thread list
	for thread in thread_list:
		thread.start()

	# Join each thread in the thread list
	'''
	Calling thread is blocked util the thread object on which it was
		called is terminated.
	'''
	for thread in thread_list:
		thread.join()

	# Restore initial values
	for i in range(len(devices)):
		devices[i].nodemap.get_node("AcquisitionMode").value\
			= initial_acquisition_mode_list[i]


if __name__ == '__main__':
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
