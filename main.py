import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from Filters import Filter

def load_image():
    global img_cv
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        display_image(img_cv, original=True)  # Exibe a imagem original
        refresh_canvas()

def display_image(img, original=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Obtém o tamanho da imagem orifinal
    img_width, img_height = img_pil.size
    
    # Redimensional a imagem para caber no canvas se for muito grande
    max_size = 500
    img_pil.thumbnail((max_size, max_size))  # Maintain aspect ratio
    img_tk = ImageTk.PhotoImage(img_pil)

    # Calcula a posição para centralizar a imagem dentro do canvas se for menor
    canvas_width, canvas_height = max_size, max_size
    x_offset = (canvas_width - img_pil.width) // 2
    y_offset = (canvas_height - img_pil.height) // 2

    if original:
        original_image_canvas.delete("all")  # Limpa a canvas
        original_image_canvas.image = img_tk  # Mantém a referência viva - garbage collection
        original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        edited_image_canvas.delete("all")  # Limapa a canvas
        edited_image_canvas.image = img_tk
        edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def apply_filter(filter_type):
    global filtered_img

    if img_cv is None:
        return
    if filter_type == "low_pass_g":
        filtered_img = Filter.gaussiano(img_cv, size=filter_size)
    elif filter_type == "low_pass_m":
        filtered_img = Filter.media_filter(img_cv, size=filter_size)
    elif filter_type == "high_pass_l":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.laplaceano_gaussiana(gray, size=filter_size)*200
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "high_pass_s":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.sobel_filter(gray, size=filter_size)
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Dilatacao":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.limiarizacao(gray)
        filtered_img = Filter.dilatacao(filtered_img)*255
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Erosao":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.limiarizacao(gray)
        filtered_img = Filter.erosao(filtered_img)*255
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Abertura":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.limiarizacao(gray)
        filtered_img = Filter.abertura(filtered_img)*255
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Fecho":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.limiarizacao(gray)
        filtered_img = Filter.fechamento(filtered_img)*255
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Limiarização(Thresholding)":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.limiarizacao(gray)*255
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Limiarização(Otsu)":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.limiarizacao_otsu(gray)*255
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Limiarização_adapt":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = Filter.limiarizacao_adapt(gray)*255
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

    display_image(filtered_img, original=False)  # Exibe a imagem editada

def refresh_canvas():
    edited_image_canvas.delete("all")  # Limpa a canvas para exibir a nova imagem

global filter_size
filter_size = 3

# Definindo a GUI
root = tk.Tk()
root.title("Image Processing App")

# Define o tamanho da janela da aplicação 1200x800
root.geometry("1085x550")

# Define a cor de fundo da janela
root.config(bg="#2e2e2e")

img_cv = None

# Cria o menu da aplicação
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Filters menu
filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)
filters_menu.add_command(label="Low Pass Filter (gaussiano)", command=lambda: apply_filter("low_pass_g"))
filters_menu.add_command(label="Low Pass Filter (media)", command=lambda: apply_filter("low_pass_m"))
filters_menu.add_command(label="High Pass Filter (sobel)", command=lambda: apply_filter("high_pass_s"))
filters_menu.add_command(label="High Pass Filter (laplaciano)", command=lambda: apply_filter("high_pass_l"))
filters_menu.add_command(label="Erosão", command=lambda: apply_filter("Erosao"))
filters_menu.add_command(label="Dilatação", command=lambda: apply_filter("Dilatacao"))
filters_menu.add_command(label="Abertura", command=lambda: apply_filter("Abertura"))
filters_menu.add_command(label="Fecho", command=lambda: apply_filter("Fecho"))
filters_menu.add_command(label="Limiarização(Thresholding)", command=lambda: apply_filter("Limiarização(Thresholding)"))
filters_menu.add_command(label="Limiarização(Otsu)", command=lambda: apply_filter("Limiarização(Otsu)"))
filters_menu.add_command(label="Limiarização(Adaptativa)", command=lambda: apply_filter("Limiarização_adapt"))

def set_size(val):
    global filter_size
    filter_size = val

# Filter size
size_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filter size", menu=size_menu)
size_menu.add_command(label="3x3",   command=lambda: set_size(3))
size_menu.add_command(label="5x5",   command=lambda: set_size(5))
size_menu.add_command(label="7x7",   command=lambda: set_size(7))
size_menu.add_command(label="9x9",   command=lambda: set_size(9))
size_menu.add_command(label="11x11", command=lambda: set_size(11))

def save_temp():
    global img_cv
    img_cv = filtered_img
    display_image(img_cv, original=True)
    refresh_canvas()

# save menu
save_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Save", menu=save_menu)
save_menu.add_command(label="Save (temporário)", command=lambda: save_temp())

# Cria a canvas para a imagem original com borda (sem background)
original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

# Cria a canvas para a imagem editada com borda (sem background)
edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()
