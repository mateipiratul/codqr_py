import tkinter as tk
import re

# GF(256) lookup tables - precomputed for efficiency
GF_EXP = [0] * 256
GF_LOG = [0] * 256

def init_galois_tables():
    """Initialize the GF(256) exponential and logarithm lookup tables"""
    x = 1
    for i in range(256):
        GF_EXP[i] = x
        # GF(256) primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
        if x & 0x80:
            x = (x << 1) ^ 0x11d
        else:
            x <<= 1

    # Calculate log table
    for i in range(255):
        GF_LOG[GF_EXP[i]] = i

def gf_multiply(x, y):
    """Multiply two numbers in GF(256)"""
    if x == 0 or y == 0:
        return 0
    return GF_EXP[(GF_LOG[x] + GF_LOG[y]) % 255]

def gf_divide(x, y):
    """Divide two numbers in GF(256)"""
    if y == 0:
        raise ValueError("Division by zero in GF(256)")
    if x == 0:
        return 0
    return GF_EXP[(GF_LOG[x] + 255 - GF_LOG[y]) % 255]

def create_empty_matrix(size):
    """Create an empty QR code matrix of given size"""
    if not isinstance(size, int) or size < 21:  # Version 1 QR code minimum size
        raise ValueError("Invalid QR code size")
    return [[0 for _ in range(size)] for _ in range(size)]

def add_finder_pattern(matrix, row, col):
    """Add a finder pattern to the QR code matrix at specified position"""
    if row < 0 or col < 0 or row + 7 > len(matrix) or col + 7 > len(matrix):
        raise ValueError("Invalid finder pattern position")
        
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

def add_timing_patterns(matrix, size):
    """Add horizontal and vertical timing patterns"""
    if size < 21:
        raise ValueError("Invalid QR code size")
        
    # Horizontal timing pattern
    for i in range(8, size - 8):
        matrix[6][i] = i % 2

    # Vertical timing pattern
    for i in range(8, size - 8):
        matrix[i][6] = i % 2
        
    return matrix

def add_alignment_pattern(matrix):
    """Add alignment pattern for Version 1 QR code"""
    pattern = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    # For Version 1 QR code, alignment pattern center is at (16, 16)
    row, col = 16, 16
    for i in range(5):
        for j in range(5):
            matrix[row + i - 2][col + j - 2] = pattern[i][j]
    return matrix

def add_dark_module(matrix):
    """Add the dark module required in QR codes"""
    matrix[14][7] = 1  # Fixed position for Version 1
    return matrix

def determine_encoding_mode(data):
    """Determine the most efficient encoding mode for the input data"""
    if not data:
        raise ValueError("Empty input data")
        
    if re.match(r'^[0-9]+$', data):
        return ('Numeric', '0001', 10)
    elif re.match(r'^[0-9A-Z $%*+\-./:]+$', data):
        return ('Alphanumeric', '0010', 9)
    else:
        return ('ASCII', '0100', 8)

def encode_mode_and_length(data):
    """Encode the mode indicator and character count"""
    mode_name, mode_indicator, count_bits = determine_encoding_mode(data)
    if len(data) >= (1 << count_bits):
        raise ValueError(f"Data too long for {mode_name} mode")
    length_binary = format(len(data), f'0{count_bits}b')
    return mode_indicator + length_binary

def generate_generator_polynomial(n_corrections):
    """Generate the generator polynomial for Reed-Solomon encoding"""
    if n_corrections <= 0:
        raise ValueError("Number of error corrections must be positive")
        
    g = [1]
    for i in range(n_corrections):
        g_new = [0] * (len(g) + 1)
        g_new[0] = g[0]
        alpha = GF_EXP[i]

        for j in range(1, len(g) + 1):
            if j < len(g):
                g_new[j] = g[j] ^ gf_multiply(g[j - 1], alpha)
            else:
                g_new[j] = gf_multiply(g[j - 1], alpha)
        g = g_new

    return g

def reed_solomon_encode(data_bits, n_corrections=10):
    """Apply Reed-Solomon encoding to the data"""
    # Pad to byte boundary
    while len(data_bits) % 8 != 0:
        data_bits += '0'

    # Convert to bytes
    data_bytes = []
    for i in range(0, len(data_bits), 8):
        byte = int(data_bits[i:i + 8], 2)
        data_bytes.append(byte)

    # Generate generator polynomial
    g = generate_generator_polynomial(n_corrections)

    # Initialize message polynomial
    message = data_bytes + [0] * n_corrections

    # Perform polynomial division
    for i in range(len(data_bytes)):
        factor = message[i]
        if factor != 0:
            for j in range(len(g)):
                message[i + j] ^= gf_multiply(g[j], factor)

    # Extract error correction bytes
    error_correction = message[-n_corrections:]

    # Convert to binary
    ec_bits = ''.join(format(byte, '08b') for byte in error_correction)

    return data_bits + ec_bits

def apply_error_correction(binary_data):
    """Apply error correction to the binary data"""
    # Ensure data length is correct for Version 1-M
    if len(binary_data) > 16 * 8:  # 16 data codewords * 8 bits
        binary_data = binary_data[:16 * 8]
    while len(binary_data) < 16 * 8:
        binary_data += '0'

    return reed_solomon_encode(binary_data, n_corrections=10)

def encode_data_with_ec(matrix, data, size):
    """Encode data with error correction and place in matrix"""
    header_bits = encode_mode_and_length(data)
    mode_name, _, _ = determine_encoding_mode(data)
    
    # Encode data based on mode
    if mode_name == 'Numeric':
        # TODO: Implement numeric encoding
        pass
    elif mode_name == 'Alphanumeric':
        # TODO: Implement alphanumeric encoding
        pass
    else:  # ASCII
        data_bits = ''.join(format(ord(c), '08b') for c in data)

    binary = header_bits + data_bits
    binary_with_ec = apply_error_correction(binary)
    
    # TODO: Implement mask pattern selection and data placement
    return matrix

def draw_qr(canvas, matrix, module_size, quiet_zone):
    """Draw the QR code on the canvas"""
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
    """Generate and display a QR code"""
    size = 21  # Version 1 QR code size
    matrix = create_empty_matrix(size)

    # Add fixed patterns
    matrix = add_finder_pattern(matrix, 0, 0)
    matrix = add_finder_pattern(matrix, 0, size - 7)
    matrix = add_finder_pattern(matrix, size - 7, 0)
    matrix = add_dark_module(matrix)
    matrix = add_timing_patterns(matrix, size)
    matrix = add_alignment_pattern(matrix)

    # Encode and place data
    data = text_input.get()
    if not data:
        raise ValueError("Input data cannot be empty")
    matrix = encode_data_with_ec(matrix, data, size)

    draw_qr(canvas, matrix, module_size, quiet_zone)

def create_ui():
    """Create the user interface"""
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
    # Initialize Galois Field tables
    init_galois_tables()
    root = create_ui()
    root.mainloop()