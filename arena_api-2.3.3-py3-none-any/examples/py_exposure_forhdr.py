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
from arena_api.buffer import BufferFactory

'''
Exposure: For High Dynamic Range
	This example demonstrates dynamically updating the exposure time in order to
	grab images appropriate for high dynamic range (or HDR) imaging. HDR images
	can be created by combining a number of images acquired at various exposure
	times. This example demonstrates grabbing three images for this purpose,
	without the actual creation of an HDR image.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "
num_images = 5
exposure_high = 100000.0
exposure_mid = 50000.0
exposure_low = 25000.0


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before raising an
		exception if it fails
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
	Store initial node values, return their values at the end
	'''
	nodes = nodemap.get_node(['TriggerMode', 'TriggerSource',
							'TriggerSelector', 'TriggerSoftware',
							'TriggerArmed', 'ExposureAuto', 'ExposureTime'])

	trigger_mode_initial = nodes['TriggerMode'].value
	trigger_source_initial = nodes['TriggerSource'].value
	trigger_selector_initial = nodes['TriggerSelector'].value
	exposure_auto_initial = nodes['ExposureAuto'].value
	exposure_time_initial = nodes['ExposureTime'].value

	return nodes, [exposure_time_initial, exposure_auto_initial,
				trigger_selector_initial, trigger_source_initial,
				trigger_mode_initial]


def trigger_software_once_armed(nodes):
	'''
	Continually check until trigger is armed. Once the trigger is armed,
		it is ready to be executed.
	'''
	trigger_armed = False

	while (trigger_armed is False):
		trigger_armed = bool(nodes['TriggerArmed'].value)

	# retrieve and execute software trigger node
	nodes['TriggerSoftware'].execute()


def acquire_hdr_images(device, nodes, initial_vals):
	'''
	demonstrates exposure configuration and acquisition for HDR imaging
	(1) Sets trigger mode
	(2) Disables automatic exposure
	(3) Sets high exposure time
	(4) Gets first image
	(5) Sets medium exposure time
	(6) Gets second image
	(7) Sets low exposure time
	(8) Gets third images
	(9) Copies images into object for later processing
	(10) Does NOT process copied images
	(11) Cleans up copied images
	'''
	'''
	Prepare trigger mode
		Enable trigger mode before starting the stream. This example uses the
		trigger to control the moment that images are taken. This ensures the
		exposure time of each image in a way that a continuous stream might have
		trouble with.
	'''
	print(f"{TAB1}Prepare trigger mode")
	nodes['TriggerSelector'].value = "FrameStart"
	nodes['TriggerMode'].value = "On"
	nodes['TriggerSource'].value = "Software"

	'''
	Disable automatic exposure
		Disable automatic exposure before starting the stream. The HDR images in
		this example require three images of varied exposures, which need to be
		set manually.
	'''
	print(f"{TAB1}Disable auto exposure")
	nodes['ExposureAuto'].value = 'Off'

	'''
	Get exposure time and software trigger nodes
		The exposure time and software trigger nodes are retrieved beforehand in
		order to check for existance, readability, and writability only once
		before the stream.
	'''
	print(f"{TAB1}Get exposure time and trigger software nodes")

	if nodes['ExposureTime'] is None or nodes['TriggerSoftware'] is None:
		raise Exception("ExposureTime or TriggerSoftware node not found")

	if (nodes['ExposureTime'].is_writable is False
	or nodes['TriggerSoftware'].is_writable is False):
		raise Exception("ExposureTime or TriggerSoftware node not writable")

	'''
	If largest exposure times is not within the exposure time range, set
		largest exposure time to max value and set the remaining exposure times
		to half the value of the state before
	'''
	if (exposure_high > nodes['ExposureTime'].max
	or exposure_low < nodes['ExposureTime'].min):

		exposure_h = nodes['ExposureTime'].max
		exposure_m = exposure_h / 2
		exposure_l = exposure_m / 2

	'''
	Setup stream values
	'''
	tl_stream_nodemap = device.tl_stream_nodemap
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	# Store HDR images for processing
	hdr_images = []

	print(f"{TAB1}Acquire {num_images} HDR images")
	device.start_stream()

	for i in range(0, num_images):
		'''
		Get high, medium, and low exposure images
			This example grabs three examples of varying exposures for later
			processing. For each image, the exposure must be set, an image must
			be triggered, and then that image must be retrieved. After the
			exposure time is changed, the setting does not take place on the
			device until after the next frame. Because of this, two images are
			retrieved, the first of which is discarded.
		'''
		print(f'{TAB2}Getting HDR image {i}')

		# High exposure time
		nodes['ExposureTime'].value = exposure_h
		trigger_software_once_armed(nodes)
		image_pre_high = device.get_buffer()
		trigger_software_once_armed(nodes)
		image_high = device.get_buffer()

		print(f"{TAB1}{TAB2}Image High Exposure {nodes['ExposureTime'].value}")

		# Medium exposure time
		nodes['ExposureTime'].value = exposure_m
		trigger_software_once_armed(nodes)
		image_pre_mid = device.get_buffer()
		trigger_software_once_armed(nodes)
		image_mid = device.get_buffer()

		print(f"{TAB1}{TAB2}Image Mid Exposure {nodes['ExposureTime'].value}")

		# Low exposure time
		nodes['ExposureTime'].value = exposure_l
		trigger_software_once_armed(nodes)
		image_pre_low = device.get_buffer()
		trigger_software_once_armed(nodes)
		image_low = device.get_buffer()

		print(f"{TAB1}{TAB2}Image Low Exposure {nodes['ExposureTime'].value}")

		'''
		Copy images for processing later
			Use the image factory to copy the images for later processing. Images
			are copied in order to requeue buffers to allow for more images to be
			retrieved from the device.
		'''
		print(f"{TAB2}Copy images for HDR processing later")

		i_high = BufferFactory.copy(image_high)
		hdr_images.append(i_high)
		i_mid = BufferFactory.copy(image_mid)
		hdr_images.append(i_mid)
		i_low = BufferFactory.copy(image_low)
		hdr_images.append(i_low)

		# Requeue buffers
		device.requeue_buffer(image_pre_high)
		device.requeue_buffer(image_high)
		device.requeue_buffer(image_pre_mid)
		device.requeue_buffer(image_mid)
		device.requeue_buffer(image_pre_low)
		device.requeue_buffer(image_low)

	device.stop_stream()

	'''
	Run HDR processing
		Once the images have been retrieved and copied, they can be processed
		into an HDR image. HDR algorithms
	'''
	print(f"{TAB1}Run HDR processing")

	# Destroy copied images after processing to prevent memory leaks
	for i in range(0, hdr_images.__len__()):
		BufferFactory.destroy(hdr_images[i])

	'''
	Return nodes to initial values
	'''
	nodes['ExposureTime'].value = initial_vals[0]
	nodes['ExposureAuto'].value = initial_vals[1]
	nodes['TriggerSelector'].value = initial_vals[2]
	nodes['TriggerSource'].value = initial_vals[3]
	nodes['TriggerMode'].value = initial_vals[4]


def example_entry_point():

	devices = create_devices_with_tries()
	device = devices[0]

	nodemap = device.nodemap
	nodes, initial_vals = store_initial(nodemap)

	acquire_hdr_images(device, nodes, initial_vals)

	system.destroy_device(device)


if __name__ == "__main__":
	print("Example Started\n")
	example_entry_point()
	print("\nExample Completed")
