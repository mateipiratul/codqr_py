import tkinter as tk
import re


def create_empty_matrix(size):
    return [[0 for _ in range(size)] for _ in range(size)]


def add_finder_pattern(matrix, row, col):
    """
    Add finder pattern with proper white separation border
    Returns coordinates of protected areas
    """
    protected_areas = set()
    
    # Add the separation border coordinates to protected areas
    for i in range(-1, 8):
        for j in range(-1, 8):
            if (0 <= row + i < len(matrix)) and (0 <= col + j < len(matrix)):
                protected_areas.add((row + i, col + j))
                matrix[row + i][col + j] = 0  # Set white first

    # Core finder pattern (7x7)
    pattern = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]

    # Add the finder pattern
    for i in range(7):
        for j in range(7):
            matrix[row + i][col + j] = pattern[i][j]
    
    return matrix, protected_areas


def add_dark_module(matrix):
    """Add the dark module at position (4,9)"""
    matrix[14][8] = 1  # Fixed position
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


def is_reserved_area(row, col, protected_areas):
    """
    Check if a position is in any reserved area including white margins
    """
    if (row, col) in protected_areas:
        return True
        
    # Check if position is in timing patterns
    if row == 6 or col == 6:
        return True
        
    # Check if position is in alignment pattern
    if (14 <= row <= 18) and (14 <= col <= 18):
        return True
        
    # Check if position is in format information areas
    if row == 8 and (col < 9 or col > 13):
        return True
    if col == 8 and (row < 9 or row > 13):
        return True
        
    return False


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
        # Initialize with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
        self.exp = [0] * 512  # Extended table for easier calculation
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= 0x11D
        # Fill extended table
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def multiply(self, x, y):
        if x == 0 or y == 0:
            return 0
        return self.exp[(self.log[x] + self.log[y]) % 255]  # Fixed: Need modulo 255

def generate_polynomial(num_error_words):
    gf = GaloisField()
    g = [1]
    for i in range(num_error_words):
        # Multiply by (x + α^i)
        temp = [0] * (len(g) + 1)
        for j in range(len(g)):
            temp[j] ^= g[j]  # Copy the existing terms
            temp[j + 1] ^= gf.multiply(g[j], gf.exp[i])  # Multiply by α^i
        g = temp
    return g

def reed_solomon_encode(data_words, num_error_words):
    if not data_words:
        raise ValueError("Data words cannot be empty")
    if num_error_words < 1:
        raise ValueError("Number of error correction words must be positive")
    if len(data_words) + num_error_words > 255:
        raise ValueError("Total message length cannot exceed 255 in GF(2^8)")
    
    gf = GaloisField()
    generator = generate_polynomial(num_error_words)
    
    # Initialize message with data followed by zeros for parity
    message = list(data_words) + [0] * num_error_words
    
    # Perform polynomial division
    for i in range(len(data_words)):
        coef = message[i]
        if coef != 0:
            for j in range(1, len(generator)):
                message[i + j] ^= gf.multiply(generator[j], coef)
    
    # Combine data words with computed parity bytes
    return data_words + message[len(data_words):]

def verify_encoding():
    # Test vector for QR Code Version 1-M
    data = [32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236]
    num_error_words = 10
    
    result = reed_solomon_encode(data, num_error_words)
    print(f"Data words: {data}")
    print(f"Encoded message: {result}")
    return result

if __name__ == "__main__":
    verify_encoding()


def generate_format_bits(mask_pattern, error_correction_level):
    """
    Generate format information bits with BCH error correction
    Args:
        mask_pattern: 0-7 (3 bits)
        error_correction_level: 0=L, 1=M, 2=Q, 3=H (2 bits)
    Returns:
        15-bit format information
    """
    # Format info = 5 bits (2 EC + 3 mask)
    format_info = (error_correction_level << 3) | mask_pattern
    
    # BCH error correction calculation
    data = format_info << 10
    g = 0b10100110111  # Generator polynomial
    
    # Polynomial division
    for i in range(10, -1, -1):
        if data & (1 << (i + 10)):
            data ^= g << (i)
    
    # Combine format info and error correction
    format_bits = (format_info << 10) | (data & 0x3FF)
    
    # XOR with mask 101010000010010 to ensure no all-zero format info
    format_bits ^= 0b101010000010010
    
    return format_bits

def add_format_information(matrix, mask_pattern, error_correction_level):
    """Add format information to QR code matrix"""
    # Generate format bits with the given mask pattern and error correction level
    format_bits = generate_format_bits(mask_pattern, error_correction_level)
    
    # Convert to binary string, ensuring it's 15 bits long
    format_str = format(format_bits, '015b')
    
    # Get the matrix size dynamically
    size = len(matrix)
    
    # Add horizontal format info (left side)
    for i in range(6):
        matrix[8][i] = int(format_str[14-i])
    matrix[8][7] = int(format_str[8])
    matrix[8][8] = int(format_str[7])
    matrix[7][8] = int(format_str[6])
    
    # Add vertical format info (top side)
    for i in range(6):
        matrix[5-i][8] = int(format_str[i])
        
    # Add horizontal format info (right side)
    for i in range(7):
        matrix[8][size-7+i] = int(format_str[14-i])
        
    return matrix

def encode_data_with_ec(matrix, data, size):
    """
    Encode data and generate error correction codes
    Returns: The binary string containing both data and error correction bits
    """
    # Convert text to binary and add mode indicator and length
    mode_indicator = encode_mode_and_length(data)
    data_bits = ''.join(format(ord(c), '08b') for c in data)
    
    # Combine all bits
    all_bits = mode_indicator + data_bits
    
    # Add padding if needed
    while len(all_bits) < 128:  # 16 bytes = 128 bits
        all_bits += '0'
    
    # Split into 8-bit chunks
    data_bytes = [int(all_bits[i:i+8], 2) for i in range(0, len(all_bits), 8)]
    
    # Generate error correction bytes
    ec_bytes = reed_solomon_encode(data_bytes, 10)  # 10 EC bytes for Version 1-M
    
    # Convert all bytes to bits
    final_bits = ''
    # First interleave data bytes
    for i in range(0, len(data_bytes), 2):
        if i + 1 < len(data_bytes):
            byte1 = format(data_bytes[i], '08b')
            byte2 = format(data_bytes[i+1], '08b')
            for j in range(8):
                final_bits += byte1[j] + byte2[j]
    
    # Then add error correction bits
    for byte in ec_bytes:
        final_bits += format(byte, '08b')
    
    return final_bits


def calculate_mask_score(matrix):
    """Calculate penalty score for a masked matrix"""
    score = 0
    size = len(matrix)
    
    # Rule 1: Five or more same-colored modules in a row
    def check_pattern_penalty(row):
        penalty = 0
        count = 1
        current = row[0]
        for i in range(1, len(row)):
            if row[i] == current:
                count += 1
            else:
                if count >= 5:
                    penalty += count - 2
                count = 1
                current = row[i]
        if count >= 5:
            penalty += count - 2
        return penalty

    # Horizontal check
    for row in matrix:
        score += check_pattern_penalty(row)
    
    # Vertical check
    for j in range(size):
        column = [matrix[i][j] for i in range(size)]
        score += check_pattern_penalty(column)
    
    # Rule 2: 2x2 blocks of same color
    for i in range(size - 1):
        for j in range(size - 1):
            if matrix[i][j] == matrix[i+1][j] == matrix[i][j+1] == matrix[i+1][j+1]:
                score += 3
    
    # Rule 3: Special patterns
    pattern1 = [1,0,1,1,1,0,1]
    pattern2 = pattern1[::-1]
    
    for i in range(size):
        for j in range(size-6):
            window = [matrix[i][j+k] for k in range(7)]
            if window == pattern1 or window == pattern2:
                score += 40
    
    # Rule 4: Balance of dark/light modules
    dark_count = sum(row.count(1) for row in matrix)
    percentage = dark_count * 100 // (size * size)
    score += 10 * min(abs(percentage - 50) // 5, 10)
    
    return score

def apply_mask(matrix, mask_pattern, protected_areas):
    """
    Apply a specific mask pattern to the QR code matrix
    
    Mask Patterns:
    0: (row + column) % 2 == 0
    1: row % 2 == 0
    2: column % 3 == 0
    3: (row + column) % 3 == 0
    4: (row // 2 + column // 3) % 2 == 0
    5: ((row * column) % 2) + ((row * column) % 3) == 0
    6: (((row * column) % 2) + ((row * column) % 3)) % 2 == 0
    7: (((row + column) % 2) + ((row * column) % 3)) % 2 == 0
    """
    size = len(matrix)
    for row in range(size):
        for col in range(size):
            # Skip reserved areas
            if not is_function_pattern(row, col, size):
                # Determine if this module should be masked based on the pattern
                should_mask = False
                if mask_pattern == 0:
                    should_mask = (row + col) % 2 == 0
                elif mask_pattern == 1:
                    should_mask = row % 2 == 0
                elif mask_pattern == 2:
                    should_mask = col % 3 == 0
                elif mask_pattern == 3:
                    should_mask = (row + col) % 3 == 0
                elif mask_pattern == 4:
                    should_mask = (row // 2 + col // 3) % 2 == 0
                elif mask_pattern == 5:
                    should_mask = ((row * col) % 2) + ((row * col) % 3) == 0
                elif mask_pattern == 6:
                    should_mask = (((row * col) % 2) + ((row * col) % 3)) % 2 == 0
                elif mask_pattern == 7:
                    should_mask = (((row + col) % 2) + ((row * col) % 3)) % 2 == 0
                
                # If this module should be masked, invert it
                if should_mask:
                    matrix[row][col] ^= 1
    
    return matrix



def is_function_pattern(i, j, size):
    """Check if position is part of function patterns"""
    # Finder patterns
    if (i < 9 and j < 9) or (i < 9 and j > size-9) or (i > size-9 and j < 9):
        return True
    # Timing patterns
    if i == 6 or j == 6:
        return True
    # Alignment pattern (Version 1 has only one at fixed position)
    if size == 21 and (i in range(16,19) and j in range(16,19)):
        return True
    return False

def find_best_mask(matrix):
    """Find mask pattern with lowest penalty score"""
    best_score = float('inf')
    best_mask = 0
    best_matrix = None
    
    for mask_pattern in range(8):
        # Create a deep copy of the matrix to avoid modifying the original
        masked_matrix = [row.copy() for row in matrix]
        
        # Apply the mask pattern
        masked_matrix = apply_mask(masked_matrix, mask_pattern, None)
        
        # Calculate the score for this masked matrix
        score = calculate_mask_score(masked_matrix)
        
        # Update best mask if this one has a lower score
        if score < best_score:
            best_score = score
            best_mask = mask_pattern
            best_matrix = masked_matrix
    
    return best_mask, best_matrix


def draw_qr(canvas, matrix, module_size, quiet_zone):
    """
    Draw QR code with proper quiet zone
    """
    canvas.delete("all")
    size = len(matrix)
    
    # First draw white background for entire area including quiet zone
    total_size = (size + 2 * quiet_zone) * module_size
    canvas.create_rectangle(0, 0, total_size, total_size, fill='white', outline='')
    
    # Then draw the actual QR code modules
    for i in range(size):
        for j in range(size):
            x1 = (j + quiet_zone) * module_size
            y1 = (i + quiet_zone) * module_size
            x2 = x1 + module_size
            y2 = y1 + module_size
            if matrix[i][j] == 1:  # Only draw black modules
                canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='')

def place_data_bits(matrix, bits, protected_areas, size):
    """
    Place data bits following the exact pattern while respecting protected areas
    """
    bit_index = 0
    
    # Process columns right to left in pairs
    for right_col in range(size - 1, 0, -2):
        left_col = right_col - 1
        
        # Skip the timing pattern column
        if right_col == 6:
            right_col = 5
            left_col = 4
        
        # Going up
        for row in range(size - 1, -1, -1):
            # Right column
            if not is_reserved_area(row, right_col, protected_areas) and bit_index < len(bits):
                matrix[row][right_col] = int(bits[bit_index])
                bit_index += 1
            
            # Left column
            if left_col >= 0 and not is_reserved_area(row, left_col, protected_areas) and bit_index < len(bits):
                matrix[row][left_col] = int(bits[bit_index])
                bit_index += 1
        
        # Going down (if not at the leftmost columns)
        if right_col > 1:
            right_col -= 2
            left_col = right_col - 1
            
            for row in range(size):
                if right_col >= 0 and not is_reserved_area(row, right_col, protected_areas) and bit_index < len(bits):
                    matrix[row][right_col] = int(bits[bit_index])
                    bit_index += 1
                
                if left_col >= 0 and not is_reserved_area(row, left_col, protected_areas) and bit_index < len(bits):
                    matrix[row][left_col] = int(bits[bit_index])
                    bit_index += 1


def generate_qr(text_input, canvas, module_size, quiet_zone):
    size = 21
    matrix = create_empty_matrix(size)
    protected_areas = set()

    # Add all fixed patterns and collect protected areas
    temp_matrix, areas = add_finder_pattern(matrix, 0, 0)
    protected_areas.update(areas)
    temp_matrix, areas = add_finder_pattern(matrix, 0, size - 7)
    protected_areas.update(areas)
    temp_matrix, areas = add_finder_pattern(matrix, size - 7, 0)
    protected_areas.update(areas)
    
    matrix = add_timing_patterns(matrix, size)
    matrix = add_alignment_pattern(matrix)
    matrix = add_dark_module(matrix)

    # Encode data and get the bit string
    data = text_input.get()
    encoded_bits = encode_data_with_ec(matrix, data, size)
    
    # Place the encoded bits in the matrix
    place_data_bits(matrix, encoded_bits, protected_areas, size)
    
    # Find the best mask pattern and apply it
    best_mask, masked_matrix = find_best_mask(matrix)
    
    # Add format information with the best mask pattern
    masked_matrix = add_format_information(masked_matrix, best_mask, 1)  # Using error correction level M (1)
    
    draw_qr(canvas, masked_matrix, module_size, quiet_zone)


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

import unittest
import re

class TestQRCodeGenerator(unittest.TestCase):
    def test_create_empty_matrix(self):
        """Test matrix creation with different sizes"""
        sizes = [5, 10, 21]
        for size in sizes:
            matrix = create_empty_matrix(size)
            self.assertEqual(len(matrix), size)
            self.assertEqual(len(matrix[0]), size)
            self.assertTrue(all(all(cell == 0 for cell in row) for row in matrix))

    def test_add_finder_pattern(self):
        """Test finder pattern addition"""
        matrix = create_empty_matrix(21)
        matrix, protected_areas = add_finder_pattern(matrix, 0, 0)
        
        # Check finder pattern core
        expected_core = [
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
                self.assertEqual(matrix[i][j], expected_core[i][j], 
                    f"Mismatch at position ({i},{j})")
        
        # Check protected areas
        self.assertTrue(len(protected_areas) > 0)

    def test_determine_encoding_mode(self):
        """Test encoding mode detection"""
        test_cases = [
            ('1234', ('Numeric', '0001', 10)),
            ('HELLO WORLD', ('Alphanumeric', '0010', 9)),
            ('Hello, World!', ('ASCII', '0100', 8))
        ]
        
        for input_text, expected in test_cases:
            result = determine_encoding_mode(input_text)
            self.assertEqual(result, expected)

    def test_reed_solomon_encode(self):
        """Test Reed-Solomon encoding"""
        # Test vector from QR Code specification
        data = [32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236]
        num_error_words = 10
        
        result = reed_solomon_encode(data, num_error_words)
        
        # Verify result length
        self.assertEqual(len(result), len(data) + num_error_words)
        
        # Verify that extra error correction bytes have been added
        self.assertTrue(all(isinstance(x, int) for x in result))

    def test_generate_format_bits(self):
        """Test format bits generation"""
        # Test different mask patterns and error correction levels
        test_cases = [
            (0, 1),  # Mask 0, Error Correction Level M
            (5, 2),  # Mask 5, Error Correction Level Q
            (7, 3)   # Mask 7, Error Correction Level H
        ]
        
        for mask_pattern, error_correction in test_cases:
            format_bits = generate_format_bits(mask_pattern, error_correction)
            
            # Check that format bits are 15 bits long
            binary_format = format(format_bits, '015b')
            self.assertEqual(len(binary_format), 15)
            
            # Verify the bits can be converted back to format info
            self.assertTrue(all(bit in '01' for bit in binary_format))

    def test_mask_patterns(self):
        """Test different mask patterns"""
        size = 21
        matrix = create_empty_matrix(size)
        
        for mask_pattern in range(8):
            # Create a copy of the matrix
            test_matrix = [row.copy() for row in matrix]
            
            # Apply mask
            masked_matrix = apply_mask(test_matrix, mask_pattern, None)
            
            # Verify masking changes some bits
            changes = sum(
                masked_matrix[i][j] != matrix[i][j] 
                for i in range(size) 
                for j in range(size) 
                if not is_function_pattern(i, j, size)
            )
            
            self.assertGreater(changes, 0, f"Mask pattern {mask_pattern} made no changes")

    def test_is_function_pattern(self):
        """Test function pattern detection"""
        size = 21
        # Test finder pattern areas
        self.assertTrue(is_function_pattern(0, 0, size))
        self.assertTrue(is_function_pattern(0, 8, size))
        self.assertTrue(is_function_pattern(8, 0, size))
        
        # Test timing patterns
        self.assertTrue(is_function_pattern(6, 10, size))
        self.assertTrue(is_function_pattern(10, 6, size))
        
        # Test alignment pattern
        self.assertTrue(is_function_pattern(17, 17, size))
        
        # Test some data areas
        self.assertFalse(is_function_pattern(10, 10, size))
        self.assertFalse(is_function_pattern(15, 15, size))

def run_tests():
    """Run all QR code generator tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQRCodeGenerator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()

if __name__ == "__main__":
    root = create_ui()
    root.mainloop()

