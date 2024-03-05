
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

from arena_api.__future__.save import Writer 
from arena_api.system import system

"""
Save: Advanced
    This example shows the capabilities of the save
    library. It shows the construction of an image parameters object
    an image writer, a custom image name pattern, checks for 
    incomplete images and saves multiple images.
"""


def time_update_function():
	'''
	This function will act like a generator.
		every time it is triggered would return the time as str in the format
		shown
	'''
	while True:
		now = datetime.now()
		yield now.strftime('%H_%M_%S_%f')


def create_devices_with_tries():
	
	'''
	just a function to let example users know that a device is needed and gives
	them a chance to connected a device instead of rasing an exception
	'''
	tries = 0
	tries_max = 5
	sleep_time_secs = 10
	while tries < tries_max:  # waits for devices for a min
		devices = system.create_device()
		if not devices:
			print(
				f'try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
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
		print(f'No device found! Please connect a device and run the '
			f'example again.')
		return


def example_entry_point():
	"""
	demonstrates saving an image
	(1) registers an image writer tag for custom naming
	(2) prepares image parameters
	(3) prepares image writer
	(4) saves the image if the buffer is complete, save the image, 
		else save the image with a I_AM_INCOMPLETE tag
	(5) requeue buffer to store the next image
	"""
	
	devices = create_devices_with_tries()
	if not devices:
		return

	device = devices[0]
	tl_stream_nodemap = device.tl_stream_nodemap
	print(f'Device used in the example:\n\t{device}')

	# Enable stream auto negotiate packet size
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

	# Enable stream packet resend
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	with device.start_stream():
		print('Stream started')

		"""
		Create an image writer
			The writer, optionally, can take width, height, and bits per pixel
			of the image(s) it would save. if these arguments are not passed
			# at run time, the first buffer passed to the Writer.save()
			# function will configure the writer to the arguments buffer's width,
			# height, and bits per pixel
		"""
		writer = Writer()

		"""
		Add a tag to use in the image name
			By default the tag 'count' is defined for the user (can be
			overwritten). A generator or a generator like function to be called
			when new image saved evaluate its new name
		"""
		time_update_generator_from_func = time_update_function()
		writer.register_tag(name='time',
							generator=time_update_generator_from_func)

		"""
		Set image name pattern
			Default name for the image is 'image_<count>.jpg' where count
			is a pre-defined tag that gets updated every time a buffer image
			is saved.

		Note:
		All tags in pattern must be registered before
		assigning the pattern to writer.pattern
		"""
		writer.pattern = 'all_images\my_image_<count>_at_<time>.jpg'

		for image_count in range(100):
			buffer = device.get_buffer()
			print(f'Image buffer {image_count} received')

			"""
			Let assume that one buffer should be saved with different name.
			Choose the condition then pass the name to save. The name
			will be used only for this save call and the future calls.
			save() would use the pattern unless another name is passed.
			"""

			if buffer.is_incomplete:
				"""
				The case here that image would saved twice. One with the
				pattern and the second copy by this condition
				"""
				writer.save(buffer, f'bad\I_AM_INCOMPLETE_{image_count}.jpg')
				print(f'Image saved {writer.saved_images[-1]}')
			else:
				"""
				save the image with the pattern defined by writer.pattern
				"""
				writer.save(buffer)
				print(f'Image saved {writer.saved_images[-1]}')

			device.requeue_buffer(buffer)
			print(f'Image buffer requeued')

	"""
	device.stop_stream() is automatically called at the end of the
	context manger scope
	"""

	# clean up ----------------------------------------------------------------

	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('WARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('Example started')
	example_entry_point()
	print('Example finished successfully')
