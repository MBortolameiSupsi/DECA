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
from datetime import datetime
from arena_api.callback import callback, callback_function
from arena_api.system import system

'''
Callback: Image Callbacks
	This example demonstrates configuring an image callback for a device. Once a
	callback is registered and the device is streaming, the user-implemented
	print_buffer function will be called. print_buffer will receive the buffer
	with the image and will display the frame id and timestamp of the image
	before returning.
'''

TAB1 = " "
TAB2 = "    "


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before raising an exception
		if it fails
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


@callback_function.device.on_buffer
def print_buffer(buffer, *args, **kwargs):
	'''
	Must have the decorator on the callback function device.on_buffer
		decorator requires buffer as its first positional parameter
	'''
	now = kwargs['now_func']()
	print(f'{TAB2}Buffer: [{buffer.width} X {buffer.height}] pixels, '
		f'TimeStamp: [{now}]')


def example_entry_point():
	'''
	demonstrates callback on buffer
	(1) Connect device, setup stream
	(2) Register the callback function using callback.register
	(3) Start the stream
	(4) As the buffers on the device get filled, the callback is triggered
	(5) Stop stream and deregister the callback before destroying the device
	'''

	devices = create_devices_with_tries()
	device = devices[0]
	print(f'{TAB1}Device used in the example:\n{TAB1}{device}')

	'''
	Setup stream values
	'''
	tl_stream_nodemap = device.tl_stream_nodemap

	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	# Register the callback on the device
	handle = callback.register(
		device, print_buffer, now_func=datetime.now)
	print(f'{TAB1}Registered \'{print_buffer.__name__}\' function '
		f'on {device}\'')

	'''
	As stream starts it will grab buffers and pass them to
		the callback
	'''
	device.start_stream(1)
	print(f'{TAB1}Stream started')

	time.sleep(5)

	device.stop_stream()
	print(f'{TAB1}Stream stopped')

	'''
	Deregister must be called before device is destroyed
	'''
	callback.deregister(handle)

	system.destroy_device(device)


if __name__ == '__main__':
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
