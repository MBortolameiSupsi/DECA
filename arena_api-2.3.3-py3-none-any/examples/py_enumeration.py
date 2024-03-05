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

'''
Enumeration: Introduction
	This example introduces device enumeration. This includes opening devices,
	searching for devices, and creating and destroying a and closing the system,
	updating and retrieving the list of device.
'''

from arena_api.system import system


'''
from arena_api.system import system: instantiates _System() there's no explicit
open for us to use
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


# Note: buffer contains an image with or without chunkdata

def list_connected_devices():

	'''
	Print device info for all connected devices
	'''

	
	'''
	system.device_infos returns a list of dicts Each dict contains a list of
	attributes for a connected device Attributes include: model, vendor, serial,
	network info, and more
	'''
	device_info = system.device_infos

	# for each device: print all attributes
	for i, info in enumerate(device_info):
		print(f"\tDevice {i}:")
		for item in device_info[i]:
			print(f"\t\t{item}: {device_info[i][item]}")


def example_entry_point():

	# Get connected devices ---------------------------------------------------
	list_connected_devices()

	# Create a device
	devices = create_devices_with_tries()

	# Clean up ----------------------------------------------------------------

	'''
	Explicit destroy: unnecessary.
		Automatically called when module closes. Also: cannot reopen, as there is
		no explicit open.
	'''
	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
