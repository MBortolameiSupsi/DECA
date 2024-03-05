from arena_api.system import system
import time

'''
Pixel Correction
    This example introduces the basics of pixel correction. 
    A single arbitrary pixel is chosen and added to the device's 
    pixel correction list. These changes are then saved to the 
    camera before being removed.
'''

TAB1 = '  '
TAB2 = '    '

'''
Pixel values
    This example does not search for pixels to correct, 
    but instead just uses arbitrary pixels to demonstrate 
    what would be done if a bad pixel were found.
'''
PIXEL_X = 333
PIXEL_Y = 444

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

def example_entry_point():

    devices = create_devices_with_tries()
    device = devices[0]
    print(f'Device used in the example:\n\t{device}')
    
    nodemap = device.nodemap
    nodes = nodemap.get_node(['DefectCorrectionEnable','DefectCorrectionCount',
    'DefectCorrectionPositionX','DefectCorrectionPositionY',
    'DefectCorrectionGetNewDefect', 'DefectCorrectionIndex', 
    'DefectCorrectionApply', 'DefectCorrectionSave', 'DefectCorrectionRemove'])

    '''
    Enable pixel correction
    '''
    print(TAB1 + "Enable pixel correction")
    nodes['DefectCorrectionEnable'] = True

    pixel_correction_count = nodes['DefectCorrectionCount'].value
    print(TAB1 + "Initial pixel correction count: " + str(pixel_correction_count))

    '''
    Add new pixel to end of correction list and get its index
        Getting a new defect automatically updates the pixel correction index.
    '''
    print(TAB1 + "Add pixel to correction list")
    nodes['DefectCorrectionGetNewDefect'].execute()

    pixel_correction_updated_index = nodes['DefectCorrectionIndex'].value
    
    '''
    Set the position of pixels to be corrected
    '''
    nodes['DefectCorrectionPositionX'] = PIXEL_X
    nodes['DefectCorrectionPositionY'] = PIXEL_Y

    print(TAB2 + "Pixel index: " + str(pixel_correction_updated_index) + " (x: " + str(nodes['DefectCorrectionPositionX']) + 
    ", y: " + str(nodes['DefectCorrectionPositionY']) + ")")

    '''
    Apply correction list
        Once the pixel corrections are applied, they will take 
        effect immediately. However, by power-cycling the camera, 
        the defect list will go back to its default values.
    '''
    print(TAB1 + "Apply correction list")
    nodes['DefectCorrectionApply'].execute()

    '''
    Save correction to camera
        Optionally write the correction to the camera to make the 
        changes persistent (the camera can still be set to default by 
        executing the DefectCorrectionRestoreDefault node)
    '''
    print(TAB1 + "Save correction to camera")
    nodes['DefectCorrectionSave'].execute()

    '''
    Remove all pixels set through this example
        The index is updated so that there are no empty indices 
        after removal regardless of position
    '''
    print(TAB1 + "Find and remove pixel from correction list")
    i = pixel_correction_updated_index

    while ( i >= pixel_correction_count):
        nodes['DefectCorrectionIndex'] = i;

        x = nodes['DefectCorrectionPositionX']
        y = nodes['DefectCorrectionPositionY']
        print(TAB2 + "Pixel index: " + str(nodes['DefectCorrectionIndex'])
        + " (x: " + str(x) + ", y:" + str(y) + ")")

        if x == PIXEL_X and y == PIXEL_Y:
            print(TAB2 + "Pixels match. Remove Pixel")
            nodes['DefectCorrectionRemove'].execute()
        else:
            print(TAB2 + "Does not match");
        
        i = i-1
    

    '''
    Clean up ----------------------------------------------------------------
        Destroy device. This call is optional and will automatically be
        called for any remaining devices when the system module is unloading.
    '''
    system.destroy_device();

if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')