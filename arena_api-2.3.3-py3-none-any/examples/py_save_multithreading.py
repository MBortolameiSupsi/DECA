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
import queue
import threading
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.__future__.save import Writer
from multiprocessing import Value

'''
Acquiring and Saving Images on Seperate Threads: Introduction
	Saving images can sometimes create a bottleneck in the image acquisition
	pipeline. By sperating saving onto a separate thread, this bottle neck can be
	avoided. This example is programmed as a simple producer-consumer problem.
'''


def create_device_with_tries():
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
			print(f'Created {len(devices)} device(s)\n')
			device = devices[0]
			return device
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def get_multiple_image_buffers(device, buffer_queue, is_more_buffers):
	'''
	Acquire thirty images and add them to the queue to be saved.
		Then send the signal that no more images are incoming, so that the other
		thread knows when to stop.
	'''
	number_of_buffers = 30

	device.start_stream(number_of_buffers)
	print(f'Stream started with {number_of_buffers} buffers')

	print(f'\tGet {number_of_buffers} buffers in a list')
	buffers = device.get_buffer(number_of_buffers)

	# Print image buffer info
	for count, buffer in enumerate(buffers):
		print(f'\t\tbuffer{count:{2}} received | '
			f'Width = {buffer.width} pxl, '
			f'Height = {buffer.height} pxl, '
			f'Pixel Format = {buffer.pixel_format.name}')
		buffer_queue.put(BufferFactory.copy(buffer))
		time.sleep(0.1)

	device.requeue_buffer(buffers)
	print(f'Requeued {number_of_buffers} buffers')

	device.stop_stream()
	print(f'Stream stopped')
	is_more_buffers.value = 0


def save_image_buffers(buffer_queue, is_more_buffers):
	'''
	Creates a writer. While there are images in the queue, or
		while images will still be added to the queue, save images until the
		queue is empty. Then wait one second before checking the earlier
		conditions. This ensures that all images will be saved.
	'''
	writer = Writer()
	count = 0
	while is_more_buffers.value or not buffer_queue.empty():
		while(not buffer_queue.empty()):
			buffer = buffer_queue.get()
			writer.save(buffer, pattern=("examples/Images/"
										f"py_save_multithreading/image_{count}.jpg"))
			print(f"Saved image {count}")
			count = count + 1
		print("Queue empty, waiting 1s")
		time.sleep(1)


def example_entry_point():
	device = create_device_with_tries()
	nodemap = device.nodemap

	tl_stream_nodemap = device.tl_stream_nodemap

	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	
	'''
	is_more_buffers: object shared between threads, used to indicate if more
	buffers will be added to the queue. The Value class can be shared between
	threads by default. It may be desirable to create a queue object with this
	variable included.
	'''
	is_more_buffers = Value('i', 1)

	
	'''
	buffer_queue: consumer-producer queue that holds buffers. Python's Queue
	class can handle access from different threads by default. Acquisition thread
	acquires images and adds to queue, while the main thread saves the images to
	disk.
	'''
	buffer_queue = queue.Queue()
	acquisition_thread = threading.Thread(target=get_multiple_image_buffers,
										args=(device, buffer_queue, is_more_buffers))
	acquisition_thread.start()
	save_image_buffers(buffer_queue, is_more_buffers)
	acquisition_thread.join()

	system.destroy_device()


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
