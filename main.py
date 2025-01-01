from tkinter import Tk, Canvas
from PIL import Image, ImageTk

# Define the size of the matrix
matrix_size = 29
pixel_size = 10  # Size of each pixel block for better visibility
image_size = matrix_size * pixel_size

# Create a white image (matrix)
img = Image.new("RGB", (image_size, image_size), "white")

# Create a Tkinter window
root = Tk()
root.title("QRCODE GENERATOR")

# Convert the PIL image to a format Tkinter understands
tk_image = ImageTk.PhotoImage(img)

# Create a Canvas widget to display the image
canvas = Canvas(root, width=image_size, height=image_size)
canvas.pack()

# Add the image to the Canvas
canvas.create_image(0, 0, anchor="nw", image=tk_image)

# Run the Tkinter event loop
root.mainloop()
