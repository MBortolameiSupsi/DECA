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
Explore: Nodes
	This example explores traversing the nodes as a tree and fundamental node
	information including display name, access mode visibility, interface type,
	and value.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''

# Choose node properties to explore
explore_access = True
explore_visibility = True
explore_type = True
explore_value = True

'''
=-=-=-=-=-=-=-=-=-
=-=- EXAMPLE -=-=-
=-=-=-=-=-=-=-=-=-
'''


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before raising
		an exception if it fails
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
			print(f'Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def explore_node(node, nodemap, depth):
	'''
	explores node
	(1) retrieves display name
	(2) retrieves accessibility
	(3) retrieves visibility
	(4) retrieves interface type
	(5) retrieves value
	'''
	# Get display name
	display_name = node.display_name

	# Retrieve accessibility
	access_mode = node.access_mode
	access_mode_str = str(access_mode)

	# Retrieve visibility
	visibility = node.visibility
	visibility_str = str(visibility)

	# Retrieve interface type
	interface_type = node.interface_type
	interface_type_str = str(interface_type)

	# Retrieve value if node is not a category or register node and is readable
	value = "-"
	if (node.is_readable and
	(node.interface_type.value != 7 and node.interface_type.value != 8)):
		value = str(node.value)

	# Check and print the desired information
	access_mode_explored = None
	visibility_explored = None
	interface_type_explored = None

	if (explore_access):
		access_mode_explored = access_mode_str

	if (explore_visibility):
		visibility_explored = visibility_str

	if (explore_type):
		interface_type_explored = interface_type_str

	if (explore_value):
		value_explored = value

	print(" " * depth, '{:<40}{:<20}{:<25}{:<30}{:}'.format(display_name,
		access_mode_explored, visibility_explored, interface_type_explored,
		value_explored))

	# Explore category node children
	if (node.interface_type.value == 8):
		children = node.features
		for val in children:
			explore_node(nodemap.get_node(val), nodemap, depth+1)


def example_entry_point(nodemap):
	# Initialize the Root node, call the explore nodes function
	node = nodemap.get_node('Root')
	print("Example Started")

	explore_node(node, nodemap, 0)

	# Cleanup
	system.destroy_device()
	print()
	print("\nExample Finished")


if __name__ == "__main__":
	devices = create_devices_with_tries()
	device = devices[0]
	nodemap = device.nodemap
	example_entry_point(nodemap)
