#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 00:33:24 2024

@author: avinashamballa
"""

import struct
import math


import struct
import math

def serialize_tree(k, d, nodes, filename):
    # Calculate the total number of nodes
    N = (k ** (d + 1) - 1) // (k - 1)
    
    # Ensure the nodes list has the correct number of elements
    assert len(nodes) == N, "Node list size does not match the expected number of nodes"
    
    # Each node requires 5 bytes: 4 bytes for data and 1 byte for the leaf flag and padding
    buffer = bytearray(5 * N)
    
    # Fill the buffer with the serialized data
    offset = 0
    for node in nodes:
        float16, uint16, is_leaf = node
        
        # Pack the float16 and uint16 as usual
        struct.pack_into('eH', buffer, offset, float16, uint16)
        
        # Pack the is_leaf bit into the next available byte
        buffer[offset + 4] = is_leaf << 7  # Shift the leaf flag to the most significant bit of the next byte
        
        offset += 5
    
    # Write the buffer to a file
    with open(filename, 'wb') as f:
        f.write(buffer)

def deserialize_tree(k, d, filename):
    # Calculate the total number of nodes
    N = (k ** (d + 1) - 1) // (k - 1)
    
    # Each node requires 5 bytes: 4 bytes for data and 1 byte for the leaf flag and padding
    buffer = bytearray(5 * N)
    
    # Read the file into a buffer
    with open(filename, 'rb') as f:
        buffer = f.read()
    
    # Initialize a list to hold the deserialized nodes
    nodes = []
    
    # Extract the serialized data from the buffer
    offset = 0
    for i in range(N):
        # Unpack the float16 and uint16
        float16, uint16 = struct.unpack_from('eH', buffer, offset)
        
        # Extract the is_leaf flag from the most significant bit of the next byte
        is_leaf = (buffer[offset + 4] >> 7) & 0x01
        
        nodes.append((float16, uint16, is_leaf))
        offset += 5
    
    return nodes


# Example usage
# k = 3 (branching factor), d = 1 (depth), and the nodes contain the payloads (float16, uint16, is_leaf)

# level order traversal since it is complete and full tree
nodes = [(1.1, 10, 0), (2.2, 20, 0), (3.3, 30, 1), (4.4, 40, 1)]
serialize_tree(3, 1, nodes, 'Dataset/tree_data.bin')
deserialized_nodes = deserialize_tree(3, 1, 'Dataset/tree_data.bin')
print(deserialized_nodes)
