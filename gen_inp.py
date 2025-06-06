import numpy as np
import h5py
import random

def generate_floorplan_data(num_samples, img_size, num_macros_range, save_path, pin_position, pin_radius):
    data = []
    num_macros_list = []
    macro_details = []
    power_density_maps = []
    switching_activity_maps = []
    thermal_conductivity_maps = []
    binary_hotspot_maps = []
    connectivity_matrices = []

    for _ in range(num_samples):
        num_macros = random.randint(*num_macros_range)
        floorplan = np.zeros((img_size, img_size), dtype=np.uint8)
        power_density = np.zeros((img_size, img_size))
        switching_activity = np.zeros((img_size, img_size))
        thermal_conductivity = np.ones((img_size, img_size))
        binary_hotspot = np.zeros((img_size, img_size), dtype=np.uint8)
        connectivity_matrix = np.zeros((num_macros, num_macros))

        macros = []
        free_spaces = [(0, 0, img_size, img_size)]

        for i in range(num_macros):
            width, height = random.randint(1, img_size // 4), random.randint(1, img_size // 4)
            x, y = place_macro_tightly_near_edges(width, height, img_size, free_spaces)
            if x is None:
                break

            floorplan[x:x + width, y:y + height] = 1
            macros.append((x, y, width, height))
            update_free_spaces_tightly(x, y, width, height, free_spaces)

            macro_power_density = (width * height) / (img_size * img_size) * 10
            macro_switching_activity = (img_size / (width * height)) * 5
            macro_thermal_conductivity = random.uniform(0.5, 1.5)

            power_density[x:x + width, y:y + height] = macro_power_density
            switching_activity[x:x + width, y:y + height] = macro_switching_activity
            thermal_conductivity[x:x + width, y:y + height] = macro_thermal_conductivity

            if (macro_power_density + macro_switching_activity + macro_thermal_conductivity) < 20:
                binary_hotspot[x:x + width, y:y + height] = 1
            else:
                binary_hotspot[x:x + width, y:y + height] = 0

        # Generating a random connectivity matrix
        for i in range(num_macros):
            for j in range(i + 1, num_macros):
                connectivity_matrix[i, j] = random.randint(0, 1)
                connectivity_matrix[j, i] = connectivity_matrix[i, j]

        data.append(floorplan)
        num_macros_list.append(num_macros)
        macro_details.append(macros)
        power_density_maps.append(power_density)
        switching_activity_maps.append(switching_activity)
        thermal_conductivity_maps.append(thermal_conductivity)
        binary_hotspot_maps.append(binary_hotspot)
        connectivity_matrices.append(connectivity_matrix)

    # Convert macro_details to a fixed-length array
    max_macro_count = max(len(m) for m in macro_details)
    padded_macro_details = np.zeros((len(macro_details), max_macro_count, 4), dtype=np.int32)
    for i, macros in enumerate(macro_details):
        for j, macro in enumerate(macros):
            padded_macro_details[i, j] = macro

    data = np.array(data)
    power_density_maps = np.array(power_density_maps)
    switching_activity_maps = np.array(switching_activity_maps)
    thermal_conductivity_maps = np.array(thermal_conductivity_maps)
    binary_hotspot_maps = np.array(binary_hotspot_maps)
    connectivity_matrices = np.array(connectivity_matrices)

    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('num_macros', data=num_macros_list)
        hf.create_dataset('power_density', data=power_density_maps)
        hf.create_dataset('switching_activity', data=switching_activity_maps)
        hf.create_dataset('thermal_conductivity', data=thermal_conductivity_maps)
        hf.create_dataset('binary_hotspot', data=binary_hotspot_maps)
        hf.create_dataset('connectivity', data=connectivity_matrices)
        hf.create_dataset('macro_details', data=padded_macro_details)

def place_macro_tightly_near_edges(width, height, img_size, free_spaces):
    """Places macros tightly near the edges of the floorplan."""
    for (fx, fy, fwidth, fheight) in free_spaces:
        if fx == 0 or fy == 0 or fx + fwidth == img_size or fy + fheight == img_size:
            if fwidth >= width and fheight >= height:
                return fx, fy
    return None, None

def update_free_spaces_tightly(x, y, width, height, free_spaces):
    """Reduces free spaces while keeping macros tightly packed."""
    new_free_spaces = []
    for (fx, fy, fwidth, fheight) in free_spaces:
        if x + width <= fx or x >= fx + fwidth or y + height <= fy or y >= fy + fheight:
            new_free_spaces.append((fx, fy, fwidth, fheight))
        else:
            if x > fx:
                new_free_spaces.append((fx, fy, x - fx, fheight))
            if y > fy:
                new_free_spaces.append((fx, fy, fwidth, y - fy))
            if x + width < fx + fwidth:
                new_free_spaces.append((x + width, fy, (fx + fwidth) - (x + width), fheight))
            if y + height < fy + fheight:
                new_free_spaces.append((fx, y + height, fwidth, (fy + fheight) - (y + height)))

    free_spaces.clear()
    free_spaces.extend(new_free_spaces)

def main():
    pin_position = (0, 16)
    pin_radius = 1
    generate_floorplan_data(num_samples=10000, img_size=32, num_macros_range=(16, 16), save_path='floorplan_data.h5', pin_position=pin_position, pin_radius=pin_radius)

if __name__ == '__main__':
    main()

