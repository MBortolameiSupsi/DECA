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

from arena_api.system import system

# for printing IP addresses
import ipaddress

'''
IP Config Manual
	This example sets persistent IP on the camera. 5 parts: 1) Persistent IP
	address to 169.254.3.2 2) Subnet mask to 255.255.0.0 3) Enables persistent IP
	4) Disables DHCP 5) Disables ARP conflict detection
'''


def example_entry_point():

	# Discover devices --------------------------------------------------------
	print('Discover devices on network')
	devices = system.create_device()
	print(f'{len(devices)} devices found')

	# Choose the first device for this example
	device = devices[0]

	'''
	Calculate IPv4 addresses:
		4 sets of eight bits for addresses can calculate integer by shifting with
		2^(8x), or using library
	'''
	new_ip_address = ipaddress.IPv4Address("169.254.3.2")
	new_subnet_mask = ipaddress.IPv4Address("255.255.0.0")

	nodemap = device.nodemap

	'''
	Find previous IPv4
	'''
	old_ip_address = \
		ipaddress.IPv4Address(nodemap.get_node("GevPersistentIPAddress").value)

	old_subnet_mask = \
		ipaddress.IPv4Address(nodemap.get_node("GevPersistentSubnetMask").value)

	# Set device configurations ------------------------------------------------

	print(f"Change persistent IP from {old_ip_address} to {new_ip_address}")
	nodemap.get_node("GevPersistentIPAddress").value = int(new_ip_address)

	print(f"Change subnet mask from {old_subnet_mask} to {new_subnet_mask}")
	nodemap.get_node("GevPersistentSubnetMask").value = int(new_subnet_mask)

	print("Enable persistent IP")
	nodemap.get_node("GevCurrentIPConfigurationPersistentIP").value = True

	print("Disable DHCP")
	nodemap.get_node("GevCurrentIPConfigurationDHCP").value = False

	print("Disable ARP conflict resolution")
	nodemap.get_node("GevPersistentARPConflictDetectionEnable").value = False


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
