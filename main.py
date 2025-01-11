import tkinter as tk

def generate_qr_code(input_string):
    """Generate a simple QR code based on the input string."""
    # Create a 29x29 matrix with white pixels (0 = white, 1 = black)
    size = 29
    qr_matrix = [[0 for _ in range(size)] for _ in range(size)]

    # Placeholder: Use the length of the input to fill part of the matrix with black pixels
    # This is where the actual QR encoding algorithm will go
    for i in range(min(len(input_string), size)):
        for j in range(size):
            qr_matrix[i][j] = 1 if (i + j) % 2 == 0 else 0

    return qr_matrix


def display_qr_code(qr_matrix):
    """Display the QR code matrix using Tkinter."""
    size = len(qr_matrix)
    cell_size = 20  # Size of each cell in pixels

    # Create a Tkinter window
    root = tk.Tk()
    root.title("QR Code")
    canvas = tk.Canvas(root, width=size * cell_size, height=size * cell_size)
    canvas.pack()

    # Draw the QR code
    for i in range(size):
        for j in range(size):
            color = "black" if qr_matrix[i][j] == 1 else "white"
            x0 = j * cell_size
            y0 = i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

    root.mainloop()


import tkinter as tk


def draw_pixel(canvas, row, col, color, cell_size=20):
    """
    Draws a single pixel (rectangle) on the canvas at the specified row and column.

    Parameters:
    - canvas: The Tkinter Canvas object.
    - row: The row index of the pixel.
    - col: The column index of the pixel.
    - color: The color of the pixel (e.g., "white" or "black").
    - cell_size: The size of each cell in the grid.
    """
    x0 = col * cell_size
    y0 = row * cell_size
    x1 = x0 + cell_size
    y1 = y0 + cell_size
    canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)


def finder_patterns(canvas, row, col, cell_size=20):
    """
    Draws a QR code Finder Pattern (7x7) on the canvas starting at the specified row and column.

    Parameters:
    - canvas: The Tkinter Canvas object.
    - row: The starting row index for the Finder Pattern.
    - col: The starting column index for the Finder Pattern.
    - cell_size: The size of each cell in the grid.
    """
    for i in range(row, row + 7):
        for j in range(col, col + 7):
            # Determine the conditions for inner horizontal and vertical borders
            is_inner_horizontal_border = (i == row + 1 or i == row + 5) and (j > col and j < col + 6)
            is_inner_vertical_border = (j == col + 1 or j == col + 5) and (i > row and i < row + 6)

            if is_inner_horizontal_border or is_inner_vertical_border:
                draw_pixel(canvas, i, j, "white", cell_size)  # White pixel
            else:
                draw_pixel(canvas, i, j, "black", cell_size)  # Black pixel

if __name__ == "__main__":
    # Accept input string from the terminal
    input_string = input("Enter the string to generate QR code: ").strip()

    # Generate the QR code matrix
    qr_matrix = generate_qr_code(input_string)

    # Display the QR code
    display_qr_code(qr_matrix)
