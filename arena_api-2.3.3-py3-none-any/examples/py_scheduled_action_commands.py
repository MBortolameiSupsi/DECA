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
Scheduled Action Commands
	This example introduces scheduling action commands on multiple cameras. The
	device settings are configured to allow each device to trigger a single image
	using action commands. The system is prepared to receive an action command
	and the devices' PTP relationships are synchronized. This allows actions
	commands to be fired across all devices, resulting in simultaneously acquired
	images with synchronized timestamps. Depending on the initial PTP state of
	each camera, it can take about 40 seconds for all devices to autonegotiate.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "
# Exposure time to set in microseconds
EXPOSURE_TIME_TO_SET_US = 500.0
# Delta time in nanoseconds to set action command
DELTA_TIME_NS = 1000000000

# Creating global system nodemap
sys_tl_map = system.tl_system_nodemap


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before raising
		an exception if it fails
	'''
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:
		devices = system.create_device()
		if not devices:
			print(
				f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs}'
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


def store_initial(device):
	dev_map = device.nodemap
	exposure_auto_initial = dev_map['ExposureAuto'].value
	trigger_source_initial = dev_map['TriggerSource'].value
	action_uncond_initial = dev_map['ActionUnconditionalMode'].value
	action_selector_initial = dev_map['ActionSelector'].value
	action_group_key_initial = dev_map['ActionGroupKey'].value
	action_group_mask_initial = dev_map['ActionGroupMask'].value
	transfer_control_mode_initial = dev_map['TransferControlMode'].value
	ptp_enable_initial = dev_map['PtpEnable'].value
	action_command_dev_key_initial = sys_tl_map['ActionCommandDeviceKey'].value
	action_command_grp_key_initial = sys_tl_map['ActionCommandGroupKey'].value
	action_command_grp_mask_initial = sys_tl_map['ActionCommandGroupMask']\
		.value
	action_command_target_ip_initial = sys_tl_map['ActionCommandTargetIP']\
		.value

	initial_vals = [
		exposure_auto_initial, trigger_source_initial, action_uncond_initial,
		action_selector_initial, action_group_key_initial,
		action_group_mask_initial, transfer_control_mode_initial,
		ptp_enable_initial, action_command_dev_key_initial,
		action_command_grp_key_initial, action_command_grp_mask_initial,
		action_command_target_ip_initial
		]
	return initial_vals


def set_autonegotiation(device):
	'''
	Use max supported packet size. We use transfer control to ensure that
		only one camera is transmitting at a time.
	'''
	device.tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True


def set_exposure_time(device):
	'''
	Manually set exposure time
		In order to get synchronized images, the exposure time must be
		synchronized.
	'''
	dev_map = device.nodemap
	nodes = dev_map.get_node(['ExposureAuto', 'ExposureTime'])

	nodes['ExposureAuto'].value = 'Off'

	exposure_time_node = nodes['ExposureTime']

	min_device_exposure_time = exposure_time_node.min
	max_device_exposure_time = exposure_time_node.max

	if (EXPOSURE_TIME_TO_SET_US >= min_device_exposure_time and
			EXPOSURE_TIME_TO_SET_US <= max_device_exposure_time):
		exposure_time_node.value = EXPOSURE_TIME_TO_SET_US
	else:
		exposure_time_node.value = min_device_exposure_time


def set_trigger(device):
	'''
	Enable trigger mode and set source to action
		To trigger a single image using action commands, trigger mode must be
		enabled, the source set to an action command, and the selector set to the
		start of a frame.
	'''
	dev_map = device.nodemap

	dev_map['TriggerMode'].value = 'On'
	dev_map['TriggerSource'].value = 'Action0'
	dev_map['TriggerSelector'].value = 'FrameStart'


def set_action_command(device):
	'''
	Prepare the device to receive an action command
		Action unconditional mode allows a camera to accept action from an
		application without write access. The device key, group key, and group
		mask must match similar settings in the system's TL node map.
	'''
	dev_map = device.nodemap

	dev_map['ActionUnconditionalMode'].value = 'On'
	dev_map['ActionSelector'].value = 0
	dev_map['ActionDeviceKey'].value = 1
	dev_map['ActionGroupKey'].value = 1
	dev_map['ActionGroupMask'].value = 1


def set_transfer_control(device):
	'''
	Enable user controlled transfer control
		Synchronized cameras will begin transmiting images at the same time. To
		avoid missing packets due to collisions, we will use transfer control to
		control when each camera transmits the image.
	'''
	dev_map = device.nodemap

	dev_map['TransferControlMode'].value = 'UserControlled'
	dev_map['TransferOperationMode'].value = 'Continuous'
	dev_map['TransferStop'].execute()


def set_ptp(device):
	'''
	Synchronize devices by enabling PTP
		Enabling PTP on multiple devices causes them to negotiate amongst
		themselves so that there is a single master device while all the rest
		become slaves. The slaves' clocks all synchronize to the master's clock.
	'''
	device.nodemap['PtpEnable'].value = True


def set_system():
	'''
	Prepare the system to broadcast an action command.
		The device key, group key, group mask, and target IP must all match
		similar settings in the devices' node maps. The target IP acts as a mask.
	'''
	sys_tl_map['ActionCommandDeviceKey'].value = 1
	sys_tl_map['ActionCommandGroupKey'].value = 1
	sys_tl_map['ActionCommandGroupMask'].value = 1
	sys_tl_map['ActionCommandTargetIP'].value = 0xFFFFFFFF  # 0.0.0.0


def restore_initial(initial_vals, device):
	dev_map = device.nodemap

	dev_map['ExposureAuto'].value = initial_vals[0]
	dev_map['TriggerSource'].value = initial_vals[1]
	dev_map['ActionUnconditionalMode'].value = initial_vals[2]
	dev_map['ActionSelector'].value = initial_vals[3]
	dev_map['ActionGroupKey'].value = initial_vals[4]
	dev_map['ActionGroupMask'].value = initial_vals[5]
	dev_map['TransferControlMode'].value = initial_vals[6]
	dev_map['PtpEnable'].value = initial_vals[7]
	sys_tl_map['ActionCommandDeviceKey'].value = initial_vals[8]
	sys_tl_map['ActionCommandGroupKey'].value = initial_vals[9]
	sys_tl_map['ActionCommandGroupMask'].value = initial_vals[10]
	sys_tl_map['ActionCommandTargetIP'].value = initial_vals[11]


def synchronize_cameras(devices):

	for device in devices:
		dev_map = device.nodemap
		dev_tl_map = device.tl_stream_nodemap

		'''
		Prepare devices, system and set initial values
		'''
		print(f'{TAB1}Setting up device {device}')

		set_autonegotiation(device)
		print(f'{TAB1}Stream Auto Negotiate Packet Size Enabled :'
			f''' {dev_tl_map['StreamAutoNegotiatePacketSize'].value}''')

		set_exposure_time(device)
		print(f'''{TAB1}Exposure Time : {dev_map['ExposureTime'].value}''')

		set_trigger(device)
		print(f'''{TAB1}Trigger Source : {dev_map['TriggerSource'].value}''')

		set_action_command(device)
		print(f'{TAB1}Action commands: prepared')

		set_transfer_control(device)
		print(f'{TAB1}Transfer Control: prepared')

		set_system()
		print(f'{TAB1}System: prepared')

		set_ptp(device)
		print(f'''{TAB1}PTP Enabled : {dev_map['PtpEnable'].value}\n''')

		time.sleep(5)

	# Check for Master/Slave
	'''
	Wait for devices to negotiate their PTP relationship
		Before starting any PTP-dependent actions, it is important to wait for
		the devices to complete their negotiation; otherwise, the devices may not
		yet be synced. Depending on the initial PTP state of each camera, it can
		take about 40 seconds for all devices to autonegotiate. Below, we wait
		for the PTP status of each device until there is only one 'Master' and
		the rest are all 'Slaves'. During the negotiation phase, multiple devices
		may initially come up as Master so we will wait until the ptp negotiation
		completes.
	'''
	print(f'{TAB1}Waiting for PTP Master/Slave negotiation. '
		f'This can take up to about 40s')

	while True:
		master_found = False
		restart_sync_check = False

		for device in devices:

			ptp_status = device.nodemap['PtpStatus'].value

			# User might uncomment this line for debugging
			
			'''
			print(f'{device} is {ptp_status}')
			'''

			# Find master
			if ptp_status == 'Master':
				if master_found:
					restart_sync_check = True
					break
				master_found = True

			# Restart check until all slaves found
			elif ptp_status != 'Slave':
				restart_sync_check = True
				break

		# A single master was found and all remaining cameras are slaves
		if not restart_sync_check and master_found:
			break

		time.sleep(1)


def schedule_action_command(devices):
	'''
	Set up timing and broadcast action command
		Action commands must be scheduled for a time in the future. This can be
		done by grabbing the PTP time from a device, adding a delta to it, and
		setting it as an action command's execution time.
	'''
	device = devices[0]

	device.nodemap['PtpDataSetLatch'].execute()
	ptp_data_set_latch_value = device.nodemap['PtpDataSetLatchValue'].value

	print(f'{TAB2}Set action command to {DELTA_TIME_NS} nanoseconds from now')

	sys_tl_map['ActionCommandExecuteTime'].value \
		= ptp_data_set_latch_value + DELTA_TIME_NS

	print(f'{TAB2}Fire action command')
	'''
	Fire action command
		Action commands are fired and broadcast to all devices, but only received
		by the devices matching desired settings.
	'''
	sys_tl_map['ActionCommandFireCommand'].execute()

	# Grab image from cameras
	for device in devices:

		# Transfer Control
		device.nodemap['TransferStart'].execute()

		buffer = device.get_buffer(timeout=2000)

		device.nodemap['TransferStop'].execute()

		print(f'{TAB1}{TAB2}Received image from {device}'
			f' | Timestamp: {buffer.timestamp_ns} ns')

		device.requeue_buffer(buffer)


def example_entry_point():
	
	'''
	// Demonstrates action commands // (1) manually sets exposure, trigger and
	action command settings // (2) prepares devices for action commands // (3)
	synchronizes devices and fire action command // (4) retrieves images with
	synchronized timestamps
	'''

	devices = create_devices_with_tries()

	initial_vals_arr = []
	for device in devices:
		# Get device stream nodemap
		tl_stream_nodemap = device.tl_stream_nodemap

		# Enable stream auto negotiate packet size
		tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

		# Enable stream packet resend
		tl_stream_nodemap['StreamPacketResendEnable'].value = True

		# Store initial values
		initial_vals = store_initial(device)
		initial_vals_arr.append(initial_vals)

	synchronize_cameras(devices)

	print(f'{TAB1}Start stream')
	for device in devices:
		device.start_stream()
	'''
	Compare timestamps
		Scheduling action commands amongst PTP synchronized devices results
		synchronized images with synchronized timestamps.
	'''
	schedule_action_command(devices)

	for i in range(0, devices.__len__()):
		restore_initial(initial_vals_arr.pop(0), devices[i])

	print(f'{TAB1}Stop stream and destroy all devices')
	system.destroy_device()


if __name__ == '__main__':
	execute = input("The PTP stage might take upto 40s without feedback "
					"-- proceed? ('y' to continue): ")

	if (execute[0] is 'y'):
		print('Example started\n')
		example_entry_point()
		print('\nExample finished successfully')
	else:
		print('\nFinished')
