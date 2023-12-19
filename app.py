
from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

from scipy.ndimage import distance_transform_edt


app = Flask(__name__)

def generate_shapes(width, height, num_shapes):
    matrix = np.zeros((height, width), dtype=int)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    for shape_id in range(1, num_shapes + 1):
        # Start with a single seed point
        while True:
            x, y = np.random.randint(width), np.random.randint(height)
            if matrix[y, x] == 0:
                matrix[y, x] = shape_id
                break

        # Determine the shape size randomly
        shape_size = np.random.randint(2, 8)

        for _ in range(shape_size):
            np.random.shuffle(directions)  # Randomize directions
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and matrix[ny, nx] == 0:
                    x, y = nx, ny  # Move to the new position
                    matrix[y, x] = shape_id
                    break

        # Ensure at least two shapes touch
        if shape_id == 1:  # First shape created
            touch_point_x, touch_point_y = x, y

        if shape_id == 2:  # Create second shape close to the first
            matrix[touch_point_y, touch_point_x] = shape_id

        # Ensure at least one shape is not a straight line
        if shape_id == 3 and shape_size > 1:
            # Add an additional point in a different direction for the third shape
            for dx, dy in directions:
                additional_x, additional_y = x + dx, y + dy
                if 0 <= additional_x < width and 0 <= additional_y < height and matrix[additional_y, additional_x] == 0:
                    matrix[additional_y, additional_x] = shape_id
                    break

    return matrix


def calculate_output_matrix(input_matrix, num_shapes):
    height, width = input_matrix.shape
    output_matrix = np.full((height, width), -1)  # Start with -1 as a default value for non-background pixels
    
    # Calculate distance maps for each shape
    distance_maps = []
    for shape_id in range(1, num_shapes + 1):
        distance_map = distance_transform_edt(input_matrix != shape_id)
        distance_maps.append(distance_map)
    
    # For each background pixel, find the closest shape
    for y in range(height):
        for x in range(width):
            if input_matrix[y, x] == 0:  # It's a background pixel
                distances = [distance_maps[shape_id-1][y, x] for shape_id in range(1, num_shapes + 1)]
                min_distance = min(distances)
                closest_shapes = [shape_id for shape_id, dist in enumerate(distances, 1) if dist == min_distance]
                
                if len(closest_shapes) == 1:
                    # Only one closest shape
                    closest_shape_id = closest_shapes[0]
                    output_matrix[y, x] = -closest_shape_id
                else:
                    # Equidistant to multiple shapes, leave it as 0 to mark as red later
                    output_matrix[y, x] = 0

    # Copy the original shapes onto the output matrix with their original IDs as negative values
    for shape_id in range(1, num_shapes + 1):
        output_matrix[input_matrix == shape_id] = -shape_id
    
    return output_matrix


def plot_matrices(input_matrix, output_matrix, num_shapes):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Input matrix visualization with grid
    axs[0].set_title('Input Matrix')
    input_mat = axs[0].matshow(input_matrix, cmap='nipy_spectral')
    axs[0].grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    axs[0].grid(which='major', visible=False)
    axs[0].set_xticks(np.arange(-0.5, input_matrix.shape[1], 1), minor=True)
    axs[0].set_yticks(np.arange(-0.5, input_matrix.shape[0], 1), minor=True)
    axs[0].tick_params(which='both', length=0)
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])

    # Output matrix visualization with grid
    unique_ids = np.unique(output_matrix)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_ids)))
    color_map = {id_: color for id_, color in zip(unique_ids, colors)}
    color_map[0] = [1, 0, 0, 1]  # Red color for equidistant points (ID 0)

    # Overlay the original shapes with unique colors on top of the output matrix
    overlay_image = np.zeros((*output_matrix.shape, 4))
    shape_colors = plt.cm.Paired(np.arange(num_shapes))
    for shape_id in range(1, num_shapes + 1):
        overlay_image[input_matrix == shape_id] = shape_colors[shape_id - 1]

    # Create the output matrix background
    for unique_id in unique_ids:
        mask = (output_matrix == unique_id) & (input_matrix == 0)
        overlay_image[mask] = color_map[unique_id]

    axs[1].set_title('Output Matrix with Overlay')
    output_mat = axs[1].matshow(overlay_image)
    axs[1].grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    axs[1].grid(which='major', visible=False)
    axs[1].set_xticks(np.arange(-0.5, output_matrix.shape[1], 1), minor=True)
    axs[1].set_yticks(np.arange(-0.5, output_matrix.shape[0], 1), minor=True)
    axs[1].tick_params(which='both', length=0)
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])

    plt.tight_layout()
    image_path = f"static/matrices/matrix2d.png"
    plt.savefig(image_path)

    # Close the figure to free up memory
    plt.close()

    return image_path

def generate_shapes_3d(width, height, depth, num_shapes):
    matrix = np.zeros((height, width, depth), dtype=int)
    directions = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]

    for shape_id in range(1, num_shapes + 1):
        # Start with a single seed point in 3D
        while True:
            x, y, z = np.random.randint(width), np.random.randint(height), np.random.randint(depth)
            if matrix[y, x, z] == 0:
                matrix[y, x, z] = shape_id
                break

        # Determine the shape size randomly, minimum size of 2
        shape_size = np.random.randint(2, 8)

        for _ in range(shape_size):
            np.random.shuffle(directions)
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth and matrix[ny, nx, nz] == 0:
                    x, y, z = nx, ny, nz
                    matrix[y, x, z] = shape_id
                    break

    return matrix

def calculate_output_matrix_3d(input_matrix, num_shapes):
    depth, height, width = input_matrix.shape
    output_matrix = np.full((height, width, depth), -1)
    
    # Calculate distance maps for each shape
    distance_maps = []
    for shape_id in range(1, num_shapes + 1):
        distance_map = distance_transform_edt(input_matrix != shape_id)
        distance_maps.append(distance_map)
    
    # For each background voxel, find the closest shape
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if input_matrix[y, x, z] == 0:
                    distances = [distance_maps[shape_id-1][y, x, z] for shape_id in range(1, num_shapes + 1)]
                    min_distance = min(distances)
                    closest_shapes = [shape_id for shape_id, dist in enumerate(distances, 1) if dist == min_distance]
                    if len(closest_shapes) == 1:
                        closest_shape_id = closest_shapes[0]
                        output_matrix[y, x, z] = -closest_shape_id
                    else:
                        output_matrix[y, x, z] = 0
    for shape_id in range(1, num_shapes + 1):
        output_matrix[input_matrix == shape_id] = -shape_id
    
    return output_matrix

def visualize_input_output_3d(input_matrix, output_matrix):
    fig = plt.figure(figsize=(16, 8))

    # Visualization for the input matrix
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Input Matrix')
    plot_3d_colored_matrix(ax1, input_matrix)

    # Visualization for the output matrix
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Output Matrix')
    plot_3d_colored_matrix(ax2, output_matrix, include_equidistant=True)

    # Save the figure to a PNG file and return the path
    image_path = "static/matrices/matrix3D.png"
    plt.savefig(image_path)
    plt.close(fig)  # Close the figure to free up memory

    return image_path

def plot_3d_colored_matrix(ax, matrix, include_equidistant=False):
    # Create the color map with a unique color for each shape ID
    shape_ids = np.unique(matrix)
    colors = plt.cm.viridis((shape_ids - shape_ids.min()) / (shape_ids.max() - shape_ids.min()))
    colors[0] = (0, 0, 0, 1)  # Black for background

    # Plot each voxel
    for shape_id in shape_ids:
        color = colors[shape_id] if shape_id != 0 else (0, 0, 0, 1)  # Black for background
        if shape_id == 0 and not include_equidistant:
            continue  # Skip background if not including equidistant points
        x, y, z = np.where(matrix == shape_id)
        ax.scatter(z, y, x, color=color, marker='o', edgecolors='w', linewidth=0.5)

    if include_equidistant:
        # Highlight equidistant points in red
        equidistant_color = (1, 0, 0, 1)
        x, y, z = np.where(matrix == 0)
        ax.scatter(z, y, x, color=equidistant_color, marker='o', edgecolors='w', linewidth=0.5)

    # Set the background color to black and remove the axis for this subplot
    ax.set_facecolor('black')
    ax.axis('off')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_matrix', methods=['POST'])
def generate_matrix():
    height = int(request.form['height'])
    width = int(request.form['width'])
    num_shapes = int(request.form['num_shapes'])

    print(num_shapes, height,width)
    
    # Generate the input matrix
    input_matrix = generate_shapes(width, height, num_shapes)
    
    # Calculate the output matrix
    output_matrix = calculate_output_matrix(input_matrix, num_shapes)
    
    # Generate a unique image path
    image_link = plot_matrices(input_matrix, output_matrix, num_shapes)
    
    # Render the Jinja2 template and pass the updated image link and num_shapes as context variables
    return render_template('index.html', image_link=image_link, num_shapes=num_shapes)

@app.route('/generate_3d_matrix', methods=['POST'])
def generate_3d_matrix():
    # Get parameters from the form
    height = int(request.form['height'])
    width = int(request.form['width'])
    depth = int(request.form['depth'])
    num_shapes = int(request.form['num_shapes'])

    # Generate the 3D matrices
    input_matrix = generate_shapes_3d(width, height, depth, num_shapes)
    output_matrix = calculate_output_matrix_3d(input_matrix, num_shapes)

    # Visualize and save the matrices to an image
    image_path = visualize_input_output_3d(input_matrix, output_matrix)
    return render_template('index.html', image_link_3d=image_path)




if __name__ == '__main__':
    app.run(debug=True)
