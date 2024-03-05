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
Callbacks: On Node Change
	This example demonstrates configuring a callback to be invoked when a node is
	invalidated. A node is invalidated when its value changes or can be
	invalidated manually. In this example, a callback is registered on
	PayloadSize. The example shows two ways to invoke a callback: first by
	changing the value of a dependent node (Height) and then by invalidating
	PayloadSize manually. Whenever the callback is triggered, the callback
	function prints the updated value of the invalidated node.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "
height_one = 256
height_two = 512


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before raising
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
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


@callback_function.node.on_update
def print_node(node, *args, **kwargs):
	'''
	Do something on callback
		This function is registered with the callback and invoked when the
		registered node is invalidated. In this case, the callback prints the
		display name and value of the invalidated node.
	'''
	print(f'{TAB1}{TAB2}Message from callback')
	print(f'{TAB2}{TAB2}{node.name} : {str(node.value)}')


def cause_callback(device):
	'''
	demonstrates callbacks invoked on node changes
	(1) registers callback on node PayloadSize
	(2) changes Height twice to invalidate PayloadSize, invoking callback
	(3) invalidates PayloadSize manually
	(4) deregisters callback
	'''
	nodemap = device.nodemap

	nodes = nodemap.get_node(["Height", "PayloadSize"])

	'''
	Get node values to return their values at end of example
	'''
	height_initial = nodes["Height"].value

	if (nodes["PayloadSize"].is_readable is False):
		raise Exception("PaytloadSize not readable")

	'''
	Register PayloadSize for callbacks
		Callbacks are registered with a node and a function. This example
		demonstrates callbacks being invoked when the node is invalidated. This
		could be when the node value changes, either manually or by the device,
		or when the node is invalidated manually.
	'''
	print(f"{TAB1}Register Callback on PayloadSize")
	handle = callback.register(nodes["PayloadSize"], print_node)

	'''
	Modify Height to invoke callback on PayloadSize
		The value of PayloadSize depends on a number of other nodes. This
		includes Height. Therefore, changing the value of Height changes the
		value of and invalidates PayloadSize, which then invokes the callback.
	'''
	print(f"{TAB2}Change Height Once")
	nodes["Height"].value = height_one

	print(f"{TAB2}Change Height Twice")
	nodes["Height"].value = height_two

	'''
	Manually invalidate PayloadSize for callback
		Apart from changing the value of a node, nodes can be invalidated
		manually by calling InvalidateNode. This also invokes the callback.
	'''
	print(f"{TAB2}Invalidate PayloadSize")
	nodes["PayloadSize"].invalidate_node()

	'''
	Deregister callback
		Failing to deregister a callback results in a memory leak. Once a
		callback has been registered, it will no longer be invoked when a node is
		invalidated
	'''
	print(f'{TAB2}Deregister Callback')
	callback.deregister(handle)

	'''
	Return node(s) to their initial values
	'''
	nodes["Height"].value = height_initial


def example_entry_point():
	'''
	Prepration and Cleanup
	'''
	devices = create_devices_with_tries()
	device = devices[0]

	cause_callback(device)

	system.destroy_device(device)


if __name__ == '__main__':
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
