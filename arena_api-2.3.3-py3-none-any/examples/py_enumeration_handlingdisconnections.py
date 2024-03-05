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
import threading
from arena_api.system import system

MAX_IMAGES = 100

'''
Enumeration: Handling Disconnections
	This example demonstrates a multi-threaded approach to handling device
	disconnections. It spawns two threads, each with a different responsibility.
	First, the acquisition thread is responsible for acquiring images when the
	device is connected. Second, the enumeration thread handles disconnections by
	reconnecting the device and notifying the acquisition thread.
'''


# Declare global variables to use later
g_device = 0
g_is_running = 0
g_device_serial = 0
cv = 0


def enumeration_thread():
	global g_device
	global g_device_serial
	global g_is_running

	found_device = True
	wait_between_searches_time = 10

	cv.acquire()
	while(g_is_running):
		'''
		While device is active: wait
		'''
		if(found_device):
			print("\tSignalling acquisition thread")
			cv.notify()
			cv.wait()
			# Device disconnected
			print("Enumeration thread active")
		system.destroy_device()

		'''
		Check for device
		'''
		cv.acquire()
		if(g_is_running):
			found_device = False

			'''
			Get device infos, check for correct serial number
			'''
			print("Checking for device")
			device_infos = system.device_infos
			for device_info in device_infos:
				if(("serial", g_device_serial) in device_info.items()):
					print("Found device")
					found_device = True

			if(found_device):
				devices = system.create_device()
				for device in devices:
					if(device.nodemap.get_node("DeviceSerialNumber").value
					== g_device_serial):
						print("Connected device")
						g_device = device
			else:
				print("Failed to find device")
				print(f"Waiting {wait_between_searches_time} seconds")
				time.sleep(wait_between_searches_time)
				print("Finished waiting")
				print()
		else:
			system.destroy_device()
			cv.release()
			print("Terminating enumeration thread")
			continue


def acquisition_thread():
	'''
	Function for the acquisition thread.
		Configure global device, then continually retrieve images. Our device
		configurations persist through disconnects. If the device is
		disconnected, the thread is locked and we signal the enumeration thread
		to reconnect it
	'''
	global g_device
	global g_is_running
	global cv

	nodemap = g_device.nodemap
	tl_stream_nodemap = g_device.tl_device_nodemap

	initial_acquisition_mode = nodemap.get_node("AcquisitionMode").value

	# Set acquisition mode to continuous
	nodemap.get_node("AcquisitionMode").value = "Continuous"

	# Get device stream nodemap
	tl_stream_nodemap = g_device.tl_stream_nodemap

	# Set buffer handling mode to "Newest First"
	tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"

	# Enable stream auto negotiate packet size
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

	# Enable stream packet resend
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	num_images = 0
	g_is_running = True

	cv.acquire()

	while(g_is_running):
		'''
		While running: lock device across threads, to prevent simultaneous read/writes
			Given timeout: device has become disconnected, we release the
			condition, destroy the device, and wait until the device is
			reconnected
		'''
		try:
			g_device.start_stream()

			while(g_is_running and num_images <= MAX_IMAGES):
				print(f"\tGet image {num_images + 1}")
				buffer = g_device.get_buffer(timeout=1000)
				num_images = num_images + 1
				g_device.requeue_buffer(buffer)

		except TimeoutError:
			print("Device disconnected")
			cv.notify()
			cv.wait()
			print("Acquisition thread active")

		if(num_images > MAX_IMAGES):
			print("Acquisition completed")
			g_is_running = False
			g_device.stop_stream()
			cv.notify()
			cv.release()


def example_entry_point():
	'''
	Create global device for acquisition,
		and store its serial number so that we can identify it later.
	'''

	global g_device
	global g_is_running
	global g_device_serial
	global cv

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
			g_device = devices[0]
			g_device_serial = g_device.nodemap.get_node("DeviceSerialNumber").value
			break
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')

	'''
	Start acquisition and enumeration threads and wait for completion
	'''

	g_is_running = True
	cv = threading.Condition()
	threads = []
	threads.append(threading.Thread(target=enumeration_thread, args=[]))

	threads[0].start()

	acquisition_thread()
	threads[0].join()


if __name__ == '__main__':

	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
