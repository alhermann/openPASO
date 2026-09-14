#!/usr/bin/env python3
"""Generate mesh files for 4C plane strain elasticity problem."""

import sys

def generate_mesh(N, output_prefix):
    """Generate a mesh file for 4C with N elements per side in x and y, and 1 element in z."""
    
    # Domain: (0,1) x (0,1) x (0, 1/8)
    # N elements in x and y, 1 element in z
    
    dx = 1.0 / N
    dy = 1.0 / N
    dz = 1.0 / 8.0
    
    # Generate nodes
    nodes = []
    node_id = 1
    
    # Nodes are ordered: i_x, i_y, i_z
    # For HEX8 elements, we need 8 nodes per element
    # Node ordering for HEX8: bottom face (counter-clockwise), top face (counter-clockwise)
    # Bottom: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
    # Top: (0,0,1), (1,0,1), (1,1,1), (0,1,1)
    
    for iz in range(2):  # 2 nodes in z direction (1 element)
        z = iz * dz
        for iy in range(N + 1):
            y = iy * dy
            for ix in range(N + 1):
                x = ix * dx
                nodes.append((node_id, x, y, z))
                node_id += 1
    
    # Write mesh file in Exodus format
    with open(output_prefix + '.e', 'w') as f:
        # Exodus header
        f.write("EXODUS II\n")
        f.write("1.0\n")
        f.write(f"{len(nodes)}\n")  # num_nodes
        f.write(f"{N * N}\n")  # num_elements
        
        # Node coordinates
        f.write("\n")
        f.write("NODES\n")
        for nid, x, y, z in nodes:
            f.write(f"{nid} {x:.15e} {y:.15e} {z:.15e}\n")
        
        # Elements
        f.write("\n")
        f.write("ELEMENTS\n")
        f.write("HEXAHEDRON\n")
        f.write(f"{N * N}\n")  # num_hex
        
        elem_id = 1
        for iz in range(1):  # 1 element in z direction
            for iy in range(N):
                for ix in range(N):
                    # Calculate node indices
                    # Bottom face nodes (iz=0)
                    n000 = ix + 1 + iy * (N + 1)  # (ix, iy, 0)
                    n100 = n000 + 1  # (ix+1, iy, 0)
                    n110 = n100 + (N + 1)  # (ix+1, iy+1, 0)
                    n010 = n000 + (N + 1)  # (ix, iy+1, 0)
                    
                    # Top face nodes (iz=1)
                    offset = (N + 1) * (N + 1)
                    n001 = n000 + offset  # (ix, iy, 1)
                    n101 = n100 + offset  # (ix+1, iy, 1)
                    n111 = n110 + offset  # (ix+1, iy+1, 1)
                    n011 = n010 + offset  # (ix, iy+1, 1)
                    
                    # HEX8 node ordering: bottom face (counter-clockwise), top face (counter-clockwise)
                    # Standard Exodus ordering: 1,2,3,4,5,6,7,8
                    # where 1-4 are bottom face, 5-8 are top face
                    f.write(f"{elem_id} {n000} {n100} {n110} {n010} {n001} {n101} {n111} {n011}\n")
                    elem_id += 1
        
        # Block information
        f.write("\n")
        f.write("QUAD\n")
        f.write("0\n")  # num_quad
        
        f.write("\n")
        f.write("BLOCKS\n")
        f.write("1\n")  # num_blocks
        f.write("1\n")  # block_id
        f.write("HEXAHEDRON\n")
        f.write(f"{N * N}\n")  # num_elements_in_block
        for eid in range(1, N * N + 1):
            f.write(f"{eid}\n")
        
        # Side sets (for boundary conditions)
        f.write("\n")
        f.write("SIDE_SETS\n")
        f.write("4\n")  # num_side_sets (x=0, x=1, y=0, y=1)
        
        # x=0 face
        f.write("1\n")  # side_set_id
        f.write("x=0\n")  # name
        f.write(f"{N}\n")  # num_elements
        for iy in range(N):
            # Element on x=0 face
            elem_idx = iy * N + 1
            f.write(f"{elem_idx}\n")
        
        # x=1 face
        f.write("2\n")  # side_set_id
        f.write("x=1\n")  # name
        f.write(f"{N}\n")  # num_elements
        for iy in range(N):
            # Element on x=1 face
            elem_idx = iy * N + N
            f.write(f"{elem_idx}\n")
        
        # y=0 face
        f.write("3\n")  # side_set_id
        f.write("y=0\n")  # name
        f.write(f"{N}\n")  # num_elements
        for ix in range(N):
            # Element on y=0 face
            elem_idx = ix + 1
            f.write(f"{elem_idx}\n")
        
        # y=1 face
        f.write("4\n")  # side_set_id
        f.write("y=1\n")  # name
        f.write(f"{N}\n")  # num_elements
        for ix in range(N):
            # Element on y=1 face
            elem_idx = (N - 1) * N + ix + 1
            f.write(f"{elem_idx}\n")
    
    # Count degrees of freedom
    num_nodes = len(nodes)
    ndof = num_nodes * 3  # 3 DOF per node (ux, uy, uz)
    
    print(f"Generated mesh with {num_nodes} nodes and {N * N} elements")
    print(f"NDOF = {ndof}")
    
    return ndof

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_mesh.py <N> <output_prefix>")
        sys.exit(1)
    
    N = int(sys.argv[1])
    output_prefix = sys.argv[2]
    
    ndof = generate_mesh(N, output_prefix)
    print(f"NDOF = {ndof}")
