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
from datetime import datetime

'''
Exposure: Introduction
	This example introduces the exposure feature. An image's exposure time refers
	to the amount of time that a device's sensor is exposed to a scene before the
	data is collected. The exposure can be handled automatically or manually.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "
num_images = 25
exposure_time = 4000.0
timeout = 2000


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before
		raising an exception if it fails
	'''
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	devices = None
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


def store_initial(nodemap):
	'''
	Stores initial node values, returns their values at the end of example
	'''
	nodes = nodemap.get_node(['ExposureAuto', 'ExposureTime'])

	exposure_auto_initial = nodes['ExposureAuto'].value
	exposure_time_initial = nodes['ExposureTime'].value
	return nodes, [exposure_auto_initial, exposure_time_initial]


def configure_exposure_acquire_images(device, nodes, initial_vals):
	'''
	demonstrates basic exposure configuration
	(1) disables automatic exposure
	(2) gets exposure node
	(3) ensures exposure above min/below max
	(4) sets exposure
	(5) acquires images
	'''
	'''
	Disable automatic exposure
		Disable automatic exposure before setting an exposure time. Automatic
		exposure controls whether the exposure time is set manually or
		automatically by the device. Setting automatic exposure to 'Off' stops
		the device from automatically updating the exposure time while streaming.
	'''
	print(f"{TAB1}Disable automatic exposure")
	nodes['ExposureAuto'].value = 'Off'

	'''
	Get exposure time node
		In order to get the exposure time maximum and minimum values, get the
		exposure time node. Failed attempts to get a node return null, so check
		that the node exists. And because we expect to set its value, check that
		the exposure time node is writable.
	'''
	print(f"{TAB1}Get exposure time node")

	if nodes['ExposureTime'] is None:
		raise Exception("Exposure Time node not found")

	if nodes['ExposureTime'].is_writable is False:
		raise Exception("Exposure Time node not writeable")

	'''
	Set exposure time
		Before setting the exposure time, check that new exposure time is not
		outside of the exposure time's acceptable range. If above the maximum or
		below the minimum, update value to be within range. Lastly, set new
		exposure time.
	'''
	if exposure_time > nodes['ExposureTime'].max:
		nodes['ExposureTime'].value = nodes['ExposureTime'].max
	elif exposure_time < nodes['ExposureTime'].min:
		nodes['ExposureTime'].value = nodes['ExposureTime'].min
	else:
		nodes['ExposureTime'].value = exposure_time

	print(f"{TAB1}Set expsoure time to {nodes['ExposureTime'].value}")

	'''
	Setup stream values
	'''
	tl_stream_nodemap = device.tl_stream_nodemap
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	print(f"{TAB1}Getting {num_images} images")

	device.start_stream()

	for i in range(0, num_images):
		buffer = device.get_buffer()

		print(f'{TAB2}Buffer {i} received | '
			f'Timestamp ({datetime.now()})')

		device.requeue_buffer(buffer)

	device.stop_stream()

	'''
	Return nodes to initial values
	'''
	nodes["ExposureTime"].value = initial_vals[1]
	nodes["ExposureAuto"].value = initial_vals[0]


def example_entry_point():
	'''
	Preparation and Cleanup
	'''
	devices = create_devices_with_tries()
	device = devices[0]
	nodemap = device.nodemap

	nodes, initial_vals = store_initial(nodemap)

	configure_exposure_acquire_images(device, nodes, initial_vals)

	system.destroy_device(device)


if __name__ == "__main__":
	print("Example Started\n")
	example_entry_point()
	print("\nExample Completed")
