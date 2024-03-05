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

from arena_api.callback import callback, callback_function
from arena_api.system import system

'''
Callbacks: Polling
	This example demonstrates configuring a callback with polling. Polling allows
	for callbacks to be invoked over time.
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


@callback_function.node.on_update
def print_node_value(node):
	'''
	Callback function: to be triggered when polled, given sufficient
		elapsed time. Requires the appropriate decorator. Specific to nodes.
	'''
	print(f"\tTemperature: {node.value}")


def example_entry_point():

	# Create a device
	devices = create_devices_with_tries()
	device = devices[0]

	print(f'Device used in the example:\n\t{device}')

	'''
	Register callback on DeviceTemperature node
		Nodes are polled through their node maps. This example demonstrates
		polling the device temperature node. It has a polling time of 1 second,
		which means that its callback will not be invoked within 1 second of the
		last time it has been polled.
	'''
	print("Registering callback")
	nodemap = device.nodemap
	node = nodemap.get_node("DeviceTemperature")
	handle = callback.register(node, print_node_value)

	'''
	Poll at regular intervals
		The callback will only be invoked if the cumulative elapsed time since
		the last callback is larger than the polling time. We calculate the
		elapsed time since the last callback, and pass it to the function
		maually, in miliseconds.
	'''

	curr_time = time.time()
	time_of_last_call = time.time()

	print("Polling at regular intervals:")
	for i in range(6):
		time.sleep(0.5)
		curr_time = time.time()
		print("\tPolled")
		nodemap.poll(int(1000 * curr_time - 1000 * time_of_last_call))
		time_of_last_call = curr_time

	print("Deregistering callback")
	callback.deregister(handle)

	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
