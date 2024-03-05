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

# Settings
number_of_buffers = 30

'''
Trigger: Overlapping Trigger on Exposure End Events
	This example demonstrates the use of trigger and exposure end events to
	minimize the amount of unused exposure time between images. This is done by
	setting the device to start exposing (or trigger) right when the last
	exposure has just finished (or exposure end event). After receiving each
	exposure end event notification, the next trigger is executed to acquire the
	next image. Once all triggers have executed, the images are retrieved from
	the output queue. This example shows that there is little time between
	triggering images, and that the exposure time is close to the time between
	triggers.
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


def example_entry_point():

	devices = create_devices_with_tries()
	device = devices[0]

	''' 
	Initialize events
    	Events nodes aren't available until the events engine has been initialized.
	'''
	device.initialize_events()

	nodemap = device.nodemap

	trigger_selector_node = nodemap.get_node("TriggerSelector")
	trigger_mode_node = nodemap.get_node("TriggerMode")
	trigger_source_node = nodemap.get_node("TriggerSource")
	trigger_overlap_node = nodemap.get_node("TriggerOverlap")
	acquisition_mode = nodemap.get_node("AcquisitionMode")
	event_selector_node = nodemap.get_node("EventSelector")
	event_notification_node = nodemap.get_node("EventNotification")
	exposure_auto_node = nodemap.get_node("ExposureAuto")

	# Store initial values
	initial_acquisition_mode = acquisition_mode.value

	initial_trigger_selector = trigger_selector_node.value
	initial_trigger_mode = trigger_mode_node.value
	initial_trigger_source = trigger_source_node.value
	initial_trigger_overlap = trigger_overlap_node.value

	initial_event_selector = event_selector_node.value
	inital_event_notification = event_notification_node.value

	initial_exposure_auto = exposure_auto_node.value

	# Configure device for image acquisition
	acquisition_mode.value = "Continuous"

	tl_stream_nodemap = device.tl_stream_nodemap

	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	'''
	Configure trigger mode for image acquisitions
	'''

	'''
	Set trigger selector
		Set the trigger selector to FrameStart. When triggered, the device will
		start acquiring a single frame. This can also be set to AcquisitionStart
		or FrameBurstStart.
	'''

	trigger_selector_node.value = "FrameStart"

	'''
	Set trigger mode
		Enable trigger mode before setting the source and selector and before
		starting the stream. Trigger mode cannot be turned on and off while the
		device is streaming.
	'''

	trigger_mode_node.value = "On"

	'''
	Set trigger source
		Set the trigger source to software in order to trigger images without the
		use of any additional hardware. Lines of the GPIO can also be used to
		trigger.
	'''

	trigger_source_node.value = "Software"

	'''
	Set trigger overlap
		Trigger overlap defines when a trigger can start accepting a new frame.
		Setting trigger to overlap with the previous frame allows the camera to
		being exposing the new frame while the camera is still reading out the
		sensor data acquired from the previous frame. std::cout << TAB1 << "Set
		trigger overlap to PreviousFrame\n";
	'''

	trigger_overlap_node.value = "PreviousFrame"

	'''
	Set event selector
		We want to trigger and wait to be notified as soon as a certain event
		occurs while making the image. Here we choose to be notified at the end
		of the exposure of an image.
	'''

	event_selector_node.value = "ExposureEnd"

	'''
	Set event notification
		After choosing which event to be notified about, enabling the
		EventNotification node will turn on the notification for the event
		selected in the EventSelector node.
	'''

	event_notification_node.value = "On"

	'''
	Retrieve exposure time
		The exposure time is similar to the time between triggering images. This
		is shown by turning off automatic exposure, retrieving the exposure time
		and converting to nanoseconds.
	'''

	exposure_auto_node.value = "Off"
	exposure_time = nodemap.get_node("ExposureTime").value
	print(f"Exposure time in nanoseconds: {exposure_time * 1000}")

	'''
	Trigger an image
		Trigger an image manually, since trigger mode is enabled. This triggers
		the camera to acquire a single image. A buffer is then filled and moved
		to the output queue, where it will wait to be retrieved.
	'''
	trigger_software_node = nodemap.get_node("TriggerSoftware")
	wait_on_event_timeout = 3000

	device.start_stream(number_of_buffers)
	for i in range(number_of_buffers):
		'''
		Trigger Armed
			Continually check until trigger is armed. Once the trigger is armed,
			it is ready to be executed.
		'''
		print("Image triggered", end="")
		is_trigger_armed = False

		while(not is_trigger_armed):
			is_trigger_armed = nodemap.get_node("TriggerArmed")

		trigger_software_node.execute()

		'''
		Wait on event
			Wait on event to process before continuing. The data is created from
			the event generation, not from waiting on it.
		'''
		device.wait_on_event(wait_on_event_timeout)
		print(" and ExposureEnd Event notification arrived")

	print(f"Grabbed {number_of_buffers} buffers")

	'''
	Retrieve images and timestamps
		Images were acquired earlier Must be retrieved from device In buffers
		that we allocated earlier
	'''
	timestamp_ns = []

	for i in range(number_of_buffers):
		print("Getting buffer")
		buffer = device.get_buffer()
		timestamp_ns.append(buffer.timestamp_ns)

		print(f"Image {i} timestamp: {timestamp_ns[i]}", end="")

		if (i > 0):
			trigger_timestamp_difference = timestamp_ns[i] - timestamp_ns[i - 1]
			print(f"({trigger_timestamp_difference} ns since last trigger)")
		else:
			print()
		device.requeue_buffer(buffer)

	device.deinitialize_events()

	device.stop_stream()

	# Restore initial values
	trigger_source_node.value = initial_trigger_source
	trigger_overlap_node.value = initial_trigger_overlap
	trigger_selector_node.value = initial_trigger_selector
	trigger_mode_node.value = initial_trigger_mode
	event_notification_node.value = inital_event_notification
	event_selector_node.value = initial_event_selector
	exposure_auto_node.value = initial_exposure_auto
	acquisition_mode.value = initial_acquisition_mode

	system.destroy_device()


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
