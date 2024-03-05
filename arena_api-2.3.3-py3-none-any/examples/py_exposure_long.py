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
Long Exposure: Introduction
	This example depicts the code to increase the maximum exposure time. By
	default, Lucid cameras are prioritized to achieve maximum frame rate.
	However, due to the high frame rate configuration, the exposure time will be
	limited as it is a dependant value. If the frame rate is 30 FPS, the maximum
	allowable exposure would be 1/30 = 0.0333 seconds = 33.3 milliseconds. So, a
	decrease in the frame rate is necessary for increasing the exposure.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "
num_images = 1


def create_devices_with_tries():
	'''
	Waits for the user to connect a device
		before raising an exception if it fails
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


def store_initial(nodemap):
	'''
	Store initial node values, return their values at the end
	'''
	nodes = nodemap.get_node(['ExposureAuto', 'ExposureTime',
							'AcquisitionFrameRateEnable',
							'AcquisitionFrameRate'])

	exposure_auto_initial = nodes['ExposureAuto'].value
	exposure_time_initial = nodes['ExposureTime'].value
	acquisition_fr_enable_initial = nodes['AcquisitionFrameRateEnable'].value
	acquisition_fr_initial = nodes['AcquisitionFrameRate'].value

	initial_vals = [exposure_time_initial, exposure_auto_initial,
					acquisition_fr_initial,
					acquisition_fr_enable_initial]

	return nodes, initial_vals


def acquire_images_long_exposure(device, nodes, initial_vals):
	'''
	demonstrates exposure: long
	(1) Set Acquisition Frame Rate Enable to true
	(2) Decrease Acquisition Frame Rate
	(3) Set Exposure Auto to OFF
	(4) Increase Exposure Time
	'''
	nodes['AcquisitionFrameRateEnable'].value = True

	'''
	For the maximum exposure, the acquisition frame rate
		is set to the lowest value allowed by the camera.
	'''
	nodes['AcquisitionFrameRate'].value = nodes['AcquisitionFrameRate'].min

	'''
	Disable automatic exposure
		Disable automatic exposure before setting an exposure time. Automatic
		exposure controls whether the exposure time is set manually or
		automatically by the device. Setting automatic exposure to 'Off' stops
		the device from automatically updating the exposure time.
	'''
	nodes['ExposureAuto'].value = 'Off'
	print(f'{TAB1}Disable Auto Exposure')

	'''
	Get exposure time node
		In order to get the exposure time maximum and minimum values, get the
		exposure time node. Failed attempts to get a node return null, so check
		that the node exists. And because we expect to set its value, check that
		the exposure time node is writable.
	'''
	if nodes['ExposureTime'] is None:
		raise Exception("ExposureTime node not found")
	if nodes['ExposureTime'].is_writable is False:
		raise Exception("ExposureTime node is not writable")

	print(f'{TAB1}Minimizing Acquisition FrameRate and Maximizing Exposure')

	print(f'{TAB2}Changed Acquisition Frame Rate from {initial_vals[2]}'
		f''' to {nodes['AcquisitionFrameRate'].value}''')

	'''
	Set exposure time to the maximum value
	'''
	nodes['ExposureTime'].value = nodes['ExposureTime'].max

	print(f'{TAB2}Changed Exposure Time from {initial_vals[0]}'
		f''' to {nodes['ExposureTime'].value}''')

	'''
	Setup stream values
	'''
	tl_stream_nodemap = device.tl_stream_nodemap
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	print(f'\n{TAB1}Getting {num_images} image(s)')

	device.start_stream()

	for i in range(num_images):
		buffer = device.get_buffer()

		print(f'{TAB2}Long Exposure Image {i} Retrieved')

		device.requeue_buffer(buffer)

	device.stop_stream()

	'''
	Return nodes to intial values
	'''
	nodes['ExposureTime'].value = initial_vals[0]
	nodes['ExposureAuto'].value = initial_vals[1]
	nodes['AcquisitionFrameRate'].value = initial_vals[2]
	nodes['AcquisitionFrameRateEnable'].value = initial_vals[3]


def example_entry_point():

	devices = create_devices_with_tries()
	device = devices[0]

	nodemap = device.nodemap

	nodes, initial_vals = store_initial(nodemap)

	acquire_images_long_exposure(device, nodes, initial_vals)

	system.destroy_device(device)


if __name__ == "__main__":
	execute = input("WARNING: Image retrieval may take over 10 seconds with"
					" no feedback -- proceed? ('y' to continue): ")

	if execute[0] is 'y':
		print("Example Started\n")
		example_entry_point()
		print("\nExample Completed")
	else:
		print("Finished")
