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
Trigger: WaitForNextLeader
	WaitForNextLeader feature uses a first packet of every incoming image to
	inform users that the camera is done integrating. This is an approximation of
	what the Exposure End event does, but it simplifies the process because we
	don't need to start a whole new event channel, and reuses data that has to be
	transmitted already for the purpose of delivering the image to the user.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "
# Waiting for first packet (leader) of next image buffer timeout
TIMEOUT_MILLISEC = 2000

# Number of image buffers to capture
NUMBER_OF_BUFFERS = 10

'''
Wait for first packet of each triggered image buffer.
	In some cases user might need to reset WaitForNextleader state. This will run
	if user set this value to False. I this case example will wait for next
	leader for each 3rd buffer.
'''
WAIT_FOR_FIRST_PACKET_OF_EVERY_IMAGE_BUFFER = True


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before
		raising an exception if it fails
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


def example_entry_point():
	'''
	demonstrates trigger configuration and use of NextLeader
	(1) sets trigger mode, source, and selector
	(2) starts stream
	(3) waits until trigger is armed
	(4) triggers image
	(5) gets image
	(6) requeues buffer
	(7) stops stream
	'''
	devices = create_devices_with_tries()
	device = devices[0]

	# Store nodes' initial values ---------------------------------------------

	'''
	Get node values that will be changed in order to return their values at
		the end of the example
	'''
	triggerSelector_initial = device.nodemap['TriggerSelector'].value
	triggerMode_initial = device.nodemap['TriggerMode'].value
	triggerSource_initial = device.nodemap['TriggerSource'].value

	'''
	Set trigger selector
		Set the trigger selector to FrameStart. When triggered, the device will
		start acquiring a single frame. This can also be set to AcquisitionStart
		or FrameBurstStart.
	'''
	print(f'{TAB1}Set trigger selector to "FrameStart"\t')
	device.nodemap['TriggerSelector'].value = 'FrameStart'

	'''
	Set trigger mode
		Trigger mode needs to be set after setting TriggerSelector and before
		starting the stream. Trigger mode cannot be turned on and off while the
		device is streaming.
	'''
	print(f'{TAB1}Enable trigger mode\t')
	device.nodemap['TriggerMode'].value = 'On'

	'''
	Set trigger source
		Set the trigger source to software in order to trigger buffers without
		the use of any additional hardware. Lines of the GPIO can also be used to
		trigger.
	'''
	print(f'{TAB1}Set trigger source to "Software"\t')
	device.nodemap['TriggerSource'].value = 'Software'

	# Enable stream auto negotiate packet size
	device.tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

	# Enable stream packet resend
	device.tl_stream_nodemap['StreamPacketResendEnable'].value = True

	'''
	Start stream
		When trigger mode is off and the acquisition mode is set to stream
		continuously, starting the stream will have the camera begin acquiring a
		steady stream of image buffers. However, with trigger mode enabled, the
		device will wait for the trigger before acquiring any.
	'''
	print(f'{TAB1}Start stream\n')
	with device.start_stream(NUMBER_OF_BUFFERS):

		for buffer in range(NUMBER_OF_BUFFERS):

			'''
			Trigger Armed
				Continually check until trigger is armed. Once the trigger is
				armed, it is ready to be executed.
			'''
			print(f'{TAB2}Wait until trigger is armed')

			'''
			Continually check until trigger is armed. Once the trigger is
				armed, it is ready to be executed.
			'''
			while not device.nodemap['TriggerArmed'].value:
				continue

			# This waits for the next leader for each triggered buffer
			if WAIT_FOR_FIRST_PACKET_OF_EVERY_IMAGE_BUFFER:
				'''
				Trigger an image buffer manually, since trigger mode
					is enabled. This triggers the camera to acquire a single
					image buffer. A buffer is then filled and moved to the output
					queue, where it will wait to be retrieved. Before the image
					buffer is sent, the exposure end event will occur. This will
					happen on every iteration
				'''
				print(f'{TAB2}Trigger Image {buffer + 1}')
				device.nodemap['TriggerSoftware'].execute()

				'''
				Wait for next leader
					This will return when the leader for the next buffer arrives
					at the host if it arrives before the timeout. Otherwise it
					will throw a timeout exception
				'''
				print(f'{TAB2}Wait for leader to arrive')
				device.wait_for_next_leader(TIMEOUT_MILLISEC)
				print(f'{TAB2}Leader has arrived for buffer {buffer + 1}')

			else:
				# This waits for the leader of every 3rd buffer.
				if buffer % 3 == 0:
					'''
					Since "wait" is not called for the other buffers,
						we call "reset" to clear the current "wait" state before
						continuing. If we do not do a "reset", then the next
						"wait" would return immediately for the last leader.
					'''
					print(f'{TAB2}Resetting WaitForNextLeader state')
					device.reset_wait_for_next_leader()

				device.nodemap['TriggerSoftware'].execute()

				if buffer % 3 == 0:
					print(f'{TAB2}Wait for leader to arrive')
					device.wait_for_next_leader(TIMEOUT_MILLISEC)
					print(f'{TAB2}Leader has arrived for buffer {buffer + 1}')
			'''
			Get image buffer
				Once a buffer has been triggered, it can be retrieved. If no
				buffer has been triggered, trying to retrieve a buffer will hang
				for the duration of the timeout and then throw an exception.
				Because the device is in a trigger mode calling "get_buffer()"
				with any non default argument (ex. "get_buffer(2)") will cause
				infinite hang. The device will be waiting for second image
				infinitely.
			'''
			buffer = device.get_buffer()

			# Print some info about the image in the buffer
			print(f'{TAB2}Image retrieved ('
				f'frame ID = {buffer.frame_id}, '
				f'timestamp (ns) = {buffer.timestamp_ns})\n')

			# Requeue the image buffer
			device.requeue_buffer(buffer)

	# Return nodes to its initial values
	device.nodemap['TriggerSource'].value = triggerSource_initial
	device.nodemap['TriggerMode'].value = triggerMode_initial
	device.nodemap['TriggerSelector'].value = triggerSelector_initial

	# Clean up ----------------------------------------------------------------

	# Destroy Device
	system.destroy_device()


if __name__ == '__main__':
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
