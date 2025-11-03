import gdspy
import matplotlib.pyplot as plt
from pathlib import Path

# Load GDS file
filename = Path('imaging/sky130_sram_1kbyte_1rw1r_32x256_8_poly.gds')
lib = gdspy.GdsLibrary(infile=filename)

# Inspect subdesigns (cells)
#print("Cells in design:", list(lib.cells.keys()))
top_cell = lib.top_level()[0]
print("Top-level cell name:", top_cell.name)


'''for ref in top_cell.references:
    print(f"Top cell references cell: {ref.ref_cell.name}")
    print(f"Reference object: {ref}")
    if ref.get_bounding_box() is not None:
        print(f"Bounding box of referenced cell {ref.ref_cell.name}: {ref.get_bounding_box()}")'''

'''for layer, polys in top_cell.get_polygons(by_spec=True).items(): 
    print(layer, len(polys))

for name, cell in lib.cells.items():
    print(f"Cell name: {name} cell object: {cell}")
    for ref in cell.references:
        print(f"Cell {name} references cell {ref.ref_cell.name}")'''

for polygon in top_cell.get_polygons():
    print(polygon)
    break