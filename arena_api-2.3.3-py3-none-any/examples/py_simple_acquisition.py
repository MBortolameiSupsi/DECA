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
Simple Acquisition
	This examples demonstrates the most basic code path of acquiring an image
	using Arena SDK. This includes device enumeration, image acquisition, and
	clean up.
'''
TAB1 = "  "
TAB2 = "    "
'''
# =-=-=-=-=-=-=-=-=-
# =-=- EXAMPLE -=-=-
# =-=-=-=-=-=-=-=-=-
'''


def update_create_devices():
	'''
	Waits for the user to connect a device before raising an
		exception if it fails
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
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def enumerate_device_and_acquire_image():
	'''
	demonstrates simplest route to acquiring an image
	(1) enumerates device
	(2) acquires image
	(3) cleans up
	'''
	print(f"{TAB1}Enumerate Device")

	devices = update_create_devices()

	# Get the device
	device = devices[0]

	# Get stream nodemap to set features before streaming
	stream_nodemap = device.tl_stream_nodemap

	# Enable stream auto negotiate packet size
	stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	# Enable stream packet resend
	stream_nodemap['StreamPacketResendEnable'].value = True

	'''
	Acquire Image
		Once a device is created, use get_buffer to acquire the image and requeue
		the buffer so that the next image can be added
	'''
	with device.start_stream():

		buffer = device.get_buffer()
		print(f"{TAB2}Acquire Image")

		# Requeue to release buffer memory
		device.requeue_buffer(buffer)

	# Clean up
	print(f"{TAB1}Clean up Arena")

	system.destroy_device()


if __name__ == '__main__':
	print('\nExample started\n')
	enumerate_device_and_acquire_image()
	print('\nExample finished successfully')
