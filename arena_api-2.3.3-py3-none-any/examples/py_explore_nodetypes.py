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

"""
Explore: Node Types
    This example explores the different properties of various node's types
    including boolean, string, enumeration, integer, and float nodes. The user
    inputs the node name that they wish to access (leaving out spacing between
    words) in order to retrieve the node properties, or inputs 'x' to exit.
    See Explore_Nodes for a complete list of nodes and their respective types.
"""

"""
 =-=-=-=-=-=-=-=-=-
 =-=- EXAMPLE -=-=-
 =-=-=-=-=-=-=-=-=-
"""


def create_devices_with_tries():
    """
    Waits for the user to connect a device before raising an exception
    if it fails
    """
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


def explore_boolean(node):
    """
    explores nodes of boolean type
    (1) retrieves value
    """
    print(f"\t\t\tValue: {node.value}")


def explore_string(node):
    """
    explores nodes of string type
    (1) retrieves value
    (2) retrieves maximum value length
    """
    print(f"\t\t\tValue: {node.value}")
    print(f"\t\t\tMaximum Length: {node.max}")


def explore_integer(node):
    """
    explores nodes of type integer
    (1) retrieves value
    (2) retrieves maximum and minimum
    (3) retrieves increment and increment mode
    (4) retrieves representation
    (5) retrieves unit
    """
    print(f"\t\t\tValue: {node.value}")
    print(f"\t\t\tMaximum Length: {node.max}, Minimum Length: {node.min}")
    print(f"\t\t\tIncrement Mode: {node.inc} ({str(node.inc_mode)})")
    print(f"\t\t\tRepresentation: {str(node.representation)}")
    print(f"\t\t\tUnit: {node.unit}")


def explore_float(node):
    """
    explores nodes of type integer
    (1) retrieves value
    (2) retrieves maximum and minimum
    (3) retrieves increment and increment mode
    (4) retrieves representation
    (5) retrieves unit
    (6) retrieves display notation
    (7) retrieves display precision
    """
    print(f"\t\t\tValue: {node.name}")
    print(f"\t\t\tValue: {node.value}")
    print(f"\t\t\tMaximum Length: {node.max}, Minimum Length: {node.min}")

    if (node.inc is not None):
        print(f"\t\t\tIncrement Mode: {node.inc} ({str(node.inc_mode)})")

    print(f"\t\t\tRepresentation: {str(node.representation)}")
    print(f"\t\t\tUnit: {node.unit}")
    print(f"\t\t\tDisplay Notation: {str(node.display_notation)}")
    print(f"\t\t\tDisplay Precision: {node.display_precision}")


def explore_enumeration(node):
    """
    explores nodes of string type
    (1) retrieves value
    (2) retrieves entries
    """
    print(f"\t\t\tCurrent Entry: {node.value}")
    print(f"\t\t\tInteger Value: {node.enumentry_nodes[node.value].int_value}")
    print(f"\t\t\tEntries: {str(node.enumentry_names)}")


def explore_nodes(nodemap):
    """
    controls node exploration
    """
    node_name = input("\tInput node name to explore ('x' to exit)\n")

    # stay in loop until exit
    while True:
        # exit manually on 'x'
        if node_name.__eq__('x'):
            print("\t\tSuccesfully exited")
            break

        # get node
        node = nodemap.get_node(str(node_name))

        # explore by type
        if (node):
            if (node.interface_type.value == 3):
                explore_boolean(node)
            elif (node.interface_type.value == 6):
                explore_string(node)
            elif (node.interface_type.value == 9):
                explore_enumeration(node)
            elif (node.interface_type.value == 2):
                explore_integer(node)
            elif (node.interface_type.value == 5):
                explore_float(node)
            else:
                print("\t\t\tType not found")

        else:
            print("\t\t\tNode not found")

        # reset input
        node_name = ""
        node_name = input("\tInput node name to explore ('x' to exit)\n")


"""
 =-=-=-=-=-=-=-=-=-
 =- PREPARATION -=-
 =- & CLEAN UP =-=-
 =-=-=-=-=-=-=-=-=-
"""


def example_entry_point():
    # prepare example
    devices = create_devices_with_tries()
    device = devices[0]

    nodemap = device.nodemap

    # run example
    print("Example Started")
    explore_nodes(nodemap)

    print()

    # clean up example
    print("Destroy Device")
    system.destroy_device(device)

    print("Example Completed")


if __name__ == "__main__":
    example_entry_point()
