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
Acquisition: Rapid Acquisition
	This example demonstrates configuring device settings in order to reduce
	bandwidth and increase framerate. This includes reducing the region of
	interest, reducing bits per pixel, increasing packet size, reducing exposure
	time, and setting a large number of buffers.
'''
TAB1 = "  "
TAB2 = "    "
'''
=-=-=-=-=-=-=-=-=-
=-=- EXAMPLE -=-=-
=-=-=-=-=-=-=-=-=-
'''


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before raising an
		exception if it fails
	'''
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	devices = None
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


def store_initial_nodes(device):
	'''
	Stores the initial values of nodes that will be modified in this example
	'''
	nodemap = device.nodemap
	nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat', 'ExposureAuto',
							'ExposureTime', 'DeviceStreamChannelPacketSize'])
	width_initial = nodes['Width'].value
	height_initial = nodes['Height'].value
	pixel_format_initial = nodes['PixelFormat'].value
	exposure_auto_initial = nodes['ExposureAuto'].value
	exposure_time_initial = nodes['ExposureTime'].value
	stream_packet_size_initial = nodes['DeviceStreamChannelPacketSize'].value

	initial_values = [width_initial, height_initial, pixel_format_initial,
					exposure_time_initial, exposure_auto_initial,
					stream_packet_size_initial]
	return nodes, initial_values


def setup(device, nodes, width, height, pixel_format, exposure_auto):
	'''
	Set features before streaming
	'''
	if nodes['Width'].is_readable and nodes['Width'].is_writable:
		nodes['Width'].value = width
	if nodes['Height'].is_readable and nodes['Height'].is_writable:
		nodes['Height'].value = height
	if nodes['PixelFormat'].is_readable and nodes['PixelFormat'].is_writable:
		nodes['PixelFormat'].value = pixel_format
	if nodes['ExposureAuto'].is_readable and nodes['ExposureAuto'].is_writable:
		nodes['ExposureAuto'].value = exposure_auto
	if nodes['ExposureTime'].is_readable and nodes['ExposureTime'].is_writable:
		nodes['ExposureTime'].value = nodes['ExposureTime'].min
	if nodes['DeviceStreamChannelPacketSize'].is_readable \
	and nodes['DeviceStreamChannelPacketSize'].is_writable:
		'''
		Set maximum stream channel packet size
			Maximizing packet size increases frame rate by reducing the amount of
			overhead required between images. This includes both extra
			header/trailer data per packet as well as extra time from
			intra-packet spacing (the time between packets). In order to grab
			images at the maximum packet size, the ethernet adapter must be
			configured appropriately: 'Jumbo packet' must be set to its maximum,
			'UDP checksum offload' must be set to 'Rx & Tx Enabled', and
			'Received Buffers' must be set to its maximum.
		'''
		stream_packet_size_max = nodes['DeviceStreamChannelPacketSize'].max
		nodes['DeviceStreamChannelPacketSize'].value = stream_packet_size_max

	else:
		raise Exception(f'Read/Write access to devices nodes not available. '
						f'Reconnecting device recommended')

	stream_nodemap = device.tl_stream_nodemap
	stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	stream_nodemap['StreamPacketResendEnable'].value = True


def return_original(nodes, initial_values):
	'''
	Returns the nodes to their initial value
	'''
	nodes['Width'].value = initial_values[0]
	nodes['Height'].value = initial_values[1]
	nodes['PixelFormat'].value = initial_values[2]
	nodes['ExposureTime'].value = initial_values[3]
	nodes['ExposureAuto'].value = initial_values[4]
	nodes['DeviceStreamChannelPacketSize'].value = initial_values[5]


def acquire_images_rapidly(device, nodes):
	'''
	demonstrates configuration for high frame rates
	(1) lowers image size
	(2) maximizes packet size
	(3) minimizes exposure time
	(4) sets high number of buffers
	(5) waits until after acquisition to requeue buffers
	'''

	'''
	Set low width and height
		Reducing the size of an image reduces the amount of bandwidth required
		for each image. The less bandwidth required per image, the more images
		can be sent over the same bandwidth.
	'''
	width = 100
	height = 100
	print(f'{TAB1}Set low width and height ({width}x{height})')

	'''
	Set small pixel format
		Similar to reducing the ROI, reducing the number of bits per pixel also
		reduces the bandwidth required for each image. The smallest pixel formats
		are 8-bit bayer and 8-bit mono (i.e. BayerRG8 and Mono8).
	'''
	pixel_format = 'Mono8'
	print(f'{TAB1}Set small pixel format ({pixel_format})')

	'''
	Set low exposure time
		Decreasing exposure time can increase frame rate by reducing the amount
		of time it takes to grab an image. Reducing the exposure time past
		certain thresholds can cause problems related to not having enough light.
		In certain situations this can be mitigated by increasing gain and/or
		environmental light.
	'''
	exposure_auto = 'Off'

	setup(device, nodes, width, height, pixel_format, exposure_auto)

	exposure_min = nodes['ExposureTime'].value
	print(f'{TAB1}Set minimum exposure time ({exposure_min})')

	'''
	Start stream with high number of buffers
		Increasing the number of buffers can increase speeds by reducing the
		amount of time taken to requeue buffers. In this example, one buffer is
		used for each image. Of course, the amount of buffers that can be used is
		limited by the amount of space in memory.
	'''
	num_buffers = 500

	'''
	Starting the stream allocates buffers, which can be passed in as
		an argument (default: 10), and begins filling them with data. Buffers
		must later be requeued to avoid memory leaks.
	'''
	with device.start_stream(num_buffers):
		print(f'{TAB1}Stream started with 500 buffers')
		""" 'device.get_buffer(arg)' returns arg number of buffers
        the buffer is in the rgb layout """
		
		print('\tGet 500 buffers')

        # Grab images --------------------------------------------------------
		buffers = device.get_buffer(num_buffers)

        # Print image buffer info
		for count, buffer in enumerate(buffers):
			print(f'\t\tbuffer{count:{2}} received | '
                  f'Width = {buffer.width} pxl, '
                  f'Height = {buffer.height} pxl, '
                  f'Pixel Format = {buffer.pixel_format.name}')
				  
		device.requeue_buffer(buffers)
		print(f'Requeued {num_buffers} buffers')


def example_entry_point():


    # Get devices
    devices = create_devices_with_tries()
    device = devices[0]

    print(f'Device used in the example:\n\t{device}')

    # Store initial values
    nodes, initial_values = store_initial_nodes(device)

    # Acquire images
    acquire_images_rapidly(device, nodes)

    # Return nodes to original value
    print('Returning nodes to their initial value')
    return_original(nodes, initial_values)

    # Stop stream
    device.stop_stream()
    print(f'Stream stopped')

    # Destroy device
    system.destroy_device()
    print('Destroyed all created devices')


if __name__ == '__main__':
    print('\nExample started\n')
    example_entry_point()
    print('\nExample finished successfully')
