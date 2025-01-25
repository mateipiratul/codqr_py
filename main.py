import tkinter as tk
import re


def create_empty_matrix(size):
    return [[0 for _ in range(size)] for _ in range(size)]


def add_finder_pattern(matrix, row, col):
    pattern = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]

    for i in range(7):
        for j in range(7):
            matrix[row + i][col + j] = pattern[i][j]
    return matrix


def add_dark_module(matrix):
    """Add the dark module at position (4,9)"""
    matrix[14][7] = 1  # Fixed position
    return matrix


def add_timing_patterns(matrix, size):
    # Horizontal timing pattern
    for i in range(8, size - 8):
        matrix[6][i] = 1 if i % 2 == 0 else 0

    # Vertical timing pattern
    for i in range(8, size - 8):
        matrix[i][6] = 1 if i % 2 == 0 else 0
    return matrix


def add_alignment_pattern(matrix):
    pattern = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    row, col = 16, 16  # Version 1 alignment pattern position
    for i in range(5):
        for j in range(5):
            matrix[row + i - 2][col + j - 2] = pattern[i][j]
    return matrix


def is_reserved_area(row, col):
    # Check if position is in finder patterns
    in_finder_top_left = row < 7 and col < 7
    in_finder_top_right = row < 7 and col > 13
    in_finder_bottom_left = row > 13 and col < 7

    # Check if position is in timing patterns
    in_timing = row == 6 or col == 6

    # Check if position is in alignment pattern
    in_alignment = (14 <= row <= 18) and (14 <= col <= 18)

    # Check if position is in format information areas
    in_format_h = row == 8 and (col < 9 or col > 13)
    in_format_v = col == 8 and (row < 9 or row > 13)

    # Check dark module
    is_dark = row == 4 and col == 9

    return (in_finder_top_left or in_finder_top_right or in_finder_bottom_left or
            in_timing or in_alignment or in_format_h or in_format_v or is_dark)


def determine_encoding_mode(data):
    """Determine the most efficient encoding mode for the data"""
    if re.match(r'^[0-9]+$', data):
        return ('Numeric', '0001', 10)

    alphanumeric_pattern = r'^[0-9A-Z $%*+\-./:]+$'
    if re.match(alphanumeric_pattern, data):
        return ('Alphanumeric', '0010', 9)

    return ('ASCII', '0100', 8)


def encode_mode_and_length(data):
    """Encode the mode indicator and character count"""
    mode_name, mode_indicator, count_bits = determine_encoding_mode(data)
    length_binary = format(len(data), f'0{count_bits}b')
    return mode_indicator + length_binary


class GaloisField:
    def __init__(self):
        # Create GF(256) lookup tables
        self.exp = [0] * 256
        self.log = [0] * 256

        x = 1
        for i in range(256):
            self.exp[i] = x
            # GF(256) primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
            if x & 0x80:
                x = (x << 1) ^ 0x11d
            else:
                x <<= 1

        # Calculate log table
        for i in range(255):
            self.log[self.exp[i]] = i

    def multiply(self, x, y):
        if x == 0 or y == 0:
            return 0
        return self.exp[(self.log[x] + self.log[y]) % 255]

    def divide(self, x, y):
        if y == 0:
            raise ValueError("Division by zero")
        if x == 0:
            return 0
        return self.exp[(self.log[x] + 255 - self.log[y]) % 255]


def generate_generator_polynomial(n_corrections, gf):
    """Generate generator polynomial for Reed-Solomon encoding"""
    g = [1]
    for i in range(n_corrections):
        # Multiply g by (x + α^i)
        g_new = [0] * (len(g) + 1)
        g_new[0] = g[0]
        alpha = gf.exp[i]

        for j in range(1, len(g) + 1):
            if j < len(g):
                g_new[j] = g[j] ^ gf.multiply(g[j - 1], alpha)
            else:
                g_new[j] = gf.multiply(g[j - 1], alpha)
        g = g_new

    return g


def reed_solomon_encode(data_bits, n_corrections=10):
    """
    Encode data using Reed-Solomon error correction
    Args:
        data_bits: String of binary data
        n_corrections: Number of error correction codewords (10 for QR Version 1-M)
    Returns:
        String of binary data + error correction bits
    """
    # Convert binary string to bytes
    # Pad with zeros to make complete bytes
    while len(data_bits) % 8 != 0:
        data_bits += '0'

    data_bytes = []
    for i in range(0, len(data_bits), 8):
        byte = int(data_bits[i:i + 8], 2)
        data_bytes.append(byte)

    # Initialize Galois Field
    gf = GaloisField()

    # Generate generator polynomial
    g = generate_generator_polynomial(n_corrections, gf)

    # Initialize message polynomial
    message = data_bytes + [0] * n_corrections

    # Perform polynomial division
    for i in range(len(data_bytes)):
        factor = message[i]
        if factor != 0:
            for j in range(len(g)):
                message[i + j] ^= gf.multiply(g[j], factor)

    # Extract error correction bytes
    error_correction = message[-n_corrections:]

    # Convert error correction bytes to binary
    ec_bits = ''
    for byte in error_correction:
        ec_bits += format(byte, '08b')

    return data_bits + ec_bits


def apply_error_correction(binary_data):
    """
    Apply Reed-Solomon error correction to QR code data
    Args:
        binary_data: String of binary data (mode + length + data)
    Returns:
        String of binary data with error correction bits appended
    """
    # For Version 1-M QR code:
    # - Total codewords: 26
    # - Error correction codewords: 10
    # - Data codewords: 16

    # Pad the data to fill complete codewords
    while len(binary_data) < 16 * 8:  # 16 data codewords * 8 bits
        binary_data += '0'

    # Truncate if longer than allowed
    binary_data = binary_data[:16 * 8]

    # Apply Reed-Solomon encoding
    encoded_data = reed_solomon_encode(binary_data, n_corrections=10)

    return encoded_data


# Modified encode_data function to include error correction
def encode_data_with_ec(matrix, data, size):
    # Get mode and length encoding
    header_bits = encode_mode_and_length(data)

    # Encode the actual data based on the mode
    mode_name, _, _ = determine_encoding_mode(data)
    data_bits = ''

    # [Previous data encoding remains the same...]

    # Combine header and data bits
    binary = header_bits + data_bits

    # Apply error correction
    binary_with_ec = apply_error_correction(binary)
    print(f"Original data bits: {binary}")
    print(f"With error correction: {binary_with_ec}")

    # Place the data in the matrix
    # [Previous data placement code remains the same...]

    return matrix


def draw_qr(canvas, matrix, module_size, quiet_zone):
    canvas.delete("all")
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            x1 = (j + quiet_zone) * module_size
            y1 = (i + quiet_zone) * module_size
            x2 = x1 + module_size
            y2 = y1 + module_size
            color = 'black' if matrix[i][j] == 1 else 'white'
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')


def generate_qr(text_input, canvas, module_size, quiet_zone):
    size = 21
    matrix = create_empty_matrix(size)

    # Add all fixed patterns
    matrix = add_finder_pattern(matrix, 0, 0)
    matrix = add_finder_pattern(matrix, 0, size - 7)
    matrix = add_finder_pattern(matrix, size - 7, 0)
    matrix = add_dark_module(matrix)
    matrix = add_timing_patterns(matrix, size)
    matrix = add_alignment_pattern(matrix)

    # Encode and place data
    data = text_input.get()
    matrix = encode_data_with_ec(matrix, data, size)

    draw_qr(canvas, matrix, module_size, quiet_zone)


def create_ui():
    root = tk.Tk()
    root.title("QR Code Generator")

    module_size = 10
    quiet_zone = 4
    size = 21

    input_frame = tk.Frame(root, pady=10)
    input_frame.pack()

    text_input = tk.Entry(input_frame, width=30)
    text_input.pack(side=tk.LEFT, padx=5)

    canvas_size = (size + 2 * quiet_zone) * module_size
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
    canvas.pack(pady=10)

    generate_btn = tk.Button(
        input_frame,
        text="Generate",
        command=lambda: generate_qr(text_input, canvas, module_size, quiet_zone)
    )
    generate_btn.pack(side=tk.LEFT)

    return root


if __name__ == "__main__":
    root = create_ui()
    root.mainloop()