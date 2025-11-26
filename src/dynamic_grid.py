import numpy as np

def create_adaptive_grid(frame_shape, boxes, base_rows=5, base_cols=5, subdivide_threshold=2):
    """
    Creates a non-uniform grid by subdividing cells with high density.

    Args:
        frame_shape (tuple): The (height, width) of the frame.
        boxes (list): A list of bounding boxes for detected people.
        base_rows (int): The number of rows in the initial coarse grid.
        base_cols (int): The number of columns in the initial coarse grid.
        subdivide_threshold (int): The number of people in a cell required to subdivide it.

    Returns:
        list: A list of grid cell coordinates [(x0, y0, x1, y1), ...].
    """
    h, w = frame_shape
    base_cell_h, base_cell_w = h / base_rows, w / base_cols

    # 1. Create a coarse base grid and count detections in each cell
    base_grid = []
    for r in range(base_rows):
        for c in range(base_cols):
            base_grid.append((int(c * base_cell_w), int(r * base_cell_h), int((c + 1) * base_cell_w), int((r + 1) * base_cell_h)))
    
    counts = np.zeros(len(base_grid), dtype=int)
    box_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]

    for cx, cy in box_centers:
        c = int(cx // base_cell_w)
        r = int(cy // base_cell_h)
        idx = r * base_cols + c
        if 0 <= idx < len(counts):
            counts[idx] += 1

    # 2. Generate the final grid by subdividing dense cells
    final_grid = []
    for i, (x0, y0, x1, y1) in enumerate(base_grid):
        if counts[i] >= subdivide_threshold:
            # Subdivide this cell into a 2x2 grid
            cell_w = (x1 - x0) / 2
            cell_h = (y1 - y0) / 2
            for r_sub in range(2):
                for c_sub in range(2):
                    sub_x0 = int(x0 + c_sub * cell_w)
                    sub_y0 = int(y0 + r_sub * cell_h)
                    sub_x1 = int(x0 + (c_sub + 1) * cell_w)
                    sub_y1 = int(y0 + (r_sub + 1) * cell_h)
                    final_grid.append((sub_x0, sub_y0, sub_x1, sub_y1))
        else:
            # Keep the original coarse cell
            final_grid.append((x0, y0, x1, y1))
            
    return final_grid