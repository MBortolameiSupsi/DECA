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
from datetime import datetime
from arena_api.callback import callback, callback_function
from arena_api.system import system


'''
Callback: Multithreaded Image Callbacks
	This example demonstrates configuring a callback within a thread. Once the
	thread has been launched, each new image is acquired and the callback is
	triggered to retrieve the image's frame ID. Once the callback function exits,
	the image buffer is requeued. After all images have been acquired, the thread
	exits and memory is cleaned up.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "
num_buffers = 25


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before raising
		an exception
	'''
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
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


@callback_function.device.on_buffer
def print_buffer(buffer, *args, **kwargs):
	'''
	Must have the decorator on the callback function
		device.on_buffer decorator requires buffer as its first parameter Buffer
		should only be accessed by a single thread at a time
	'''
	with threading.Lock():
		print(f'{TAB2}{TAB1}Buffer callback triggered'
			f'(frame id is {buffer.frame_id})')


def get_multiple_image_buffers(device):
	'''
	Grabbing buffers from device will trigger the callback
		where the buffer information can then be safely printed
	'''
	print(f"\n{TAB1}Stream Started")
	device.start_stream(1)

	print(f'{TAB1}Getting {num_buffers} buffer(s)')

	for i in range(num_buffers):
		'''
		As buffer is retreived, the callback is triggered
		'''
		print(f'{TAB2}Buffer Retrieved')
		buffer = device.get_buffer()

		device.requeue_buffer(buffer)

	device.stop_stream()
	print(f"{TAB1}Stream Stopped")


def example_entry_point():
	'''
	demonstrates callback on buffer: multithreading
	(1) Configure pre stream nodes for all devices
	(2) Initialize handle and threads on get_multiple_image_buffers
		for all devices
	(3) Start all threads in the list
	(4) Join all threads in the list
	(5) Joining threads starts the get_multiple_image_buffers function,
		which in turn triggers callbacks, after stream is started
		and buffers are received
	(6) Deregister all handles in the list before destroying device
	'''
	devices = create_devices_with_tries()
	device = devices[0]
	print(f'{TAB1}Device used in the example:\n{TAB1}{device}')

	thread_list = []
	handle_list = []

	for device in devices:
		'''
		Setup stream values
		'''
		tl_stream_nodemap = device.tl_stream_nodemap

		tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
		tl_stream_nodemap['StreamPacketResendEnable'].value = True

	for device in devices:
		'''
		Register callback handles and initialize threads
		'''
		handle = callback.register(device, print_buffer)

		print(f'{TAB1}Registered \'{print_buffer.__name__}\' function '
			f'on {device}\'')

		thread = threading.Thread(target=get_multiple_image_buffers,
								args=(device,))

		handle_list.append(handle)
		thread_list.append(thread)

	'''
	Start and join all threads in the thread list
	'''
	for thread in thread_list:
		thread.start()

	for thread in thread_list:
		thread.join()

	'''
	Deregister each handle in the handle list
		Must be called before device is destroyed
	'''
	for handle in handle_list:
		callback.deregister(handle)

	system.destroy_device(devices)


if __name__ == '__main__':
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
