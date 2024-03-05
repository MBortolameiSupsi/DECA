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

from pprint import pprint

from arena_api.system import system

'''
Force IP
	This example demonstrates how to force network settings. It does this by
	adding 1 to the final octet of the IP address. It leaves the subnet mask and
	default gateway as is although the same method is used to change these as
	well.
'''


def add_one_to_ip(ip):

	octet0, octet1, octet2, octet3 = ip.split('.')
	if octet3 == '254':  # Avoid 255
		octet3 = '1'
	else:
		octet3 = str(int(octet3) + 1)
	return f'{octet0}.{octet1}.{octet2}.{octet3}'


def example_entry_point():

	# Discover devices --------------------------------------------------------

	print('Discover devices on network')
	device_infos = system.device_infos
	print(f'{len(device_infos)} devices found')

	if not device_infos:
		raise BaseException('No device is found!')

	# Choose the first device for this example
	device_info = device_infos[0]

	'''
	Forcing the IP address requires a device's MAC address to specify the
		device. This example grabs the IP address, subnet mask, and default
		gateway as well to display changes and return the device to its original
		IP address.
	'''
	print('Device 0 info: ')
	pprint(device_info, indent=4)

	# Create new IP -----------------------------------------------------------

	print('Current IP = ', device_info['ip'])
	new_ip = add_one_to_ip(device_info['ip'])

	# The original device_info can be used however system.force_ip
	# will used on 'mac' ,'ip' ,'subnetmask' , and 'defaultgateway'.
	# This new dictionary to show what is needed by 'System.force_ip()'
	device_info_new = {
		'mac': device_info['mac'],
		'ip': new_ip,
		'subnetmask': device_info['subnetmask'],
		'defaultgateway': device_info['defaultgateway']
	}
	print('New IP = ', device_info_new['ip'])

	# Force IP ----------------------------------------------------------------

	'''
	Note: The force_ip function can also take a list of device infos to
		force new IP addesses for multiple devices.
	'''
	print('New IP is being forced')
	system.force_ip(device_info_new)
	print('New IP was forced successfully')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
