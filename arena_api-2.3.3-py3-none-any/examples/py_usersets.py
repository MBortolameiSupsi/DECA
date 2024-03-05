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
User sets
	This example introduces user sets, a feature which allows for the saving and
	loading of multiple configurations. The example starts by changing two
	features, width and height. The device configuration is then saved to user
	set 1. The default user set is then loaded, followed by user set 1 again to
	demonstrate the the device configuration changing back and forth.
'''


def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''

	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Waits for devices
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
			return devices[0]
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def example_entry_point():
	device = create_devices_with_tries()
	nodemap = device.nodemap
	WIDTH = 576
	HEIGHT = 512

	nodemap.get_node("Width").value = WIDTH
	nodemap.get_node("Height").value = HEIGHT

	'''
	Save to user set 1
		Saving the user set saves the new width and height values. Saving a user
		set involves selecting the user set to save to on the selector node and
		then executing the save on the command node.
	'''

	'''
	Save configurations to UserSet1
	'''
	print("Saved configurations to UserSet1")
	user_set_selector_node = nodemap.get_node("UserSetSelector")
	user_set_selector_node.value = "UserSet1"
	nodemap.get_node("UserSetSave").execute()

	'''
	Display Height and Width for UserSet1
	'''
	print(f"Height: {nodemap.get_node('Height').value}")
	print(f"Width: {nodemap.get_node('Width').value}")
	print()

	'''
	Load and display Height and Width for Default user set
	'''
	print("Loaded Default user set")
	user_set_selector_node.value = "Default"
	nodemap.get_node("UserSetLoad").execute()
	print(f"Height: {nodemap.get_node('Height').value}")
	print(f"Width: {nodemap.get_node('Width').value}")
	print()


	'''
	Reload and display Height and Width for UserSet1
	'''
	print("Loaded UserSet1")
	user_set_selector_node.value = "UserSet1"
	nodemap.get_node("UserSetLoad").execute()
	print(f"Height: {nodemap.get_node('Height').value}")
	print(f"Width: {nodemap.get_node('Width').value}")
	print()

	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
