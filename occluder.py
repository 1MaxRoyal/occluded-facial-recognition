import win32gui
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import cv2 as cv
import numpy as np
from ctypes import windll

# Make program aware of DPI scaling
user32 = windll.user32
user32.SetProcessDPIAware()

def occlude(img):
    app = tk.Tk()
    app.geometry("500x500")

    canvas = tk.Canvas(app, bg='black')
    canvas.pack(anchor='nw', fill='both', expand=1)
    
    def get_x_y(event):
        global x, y
        x, y = event.x, event.y
        
    def draw(event):
        global x, y
        canvas.create_line((x, y, event.x, event.y), 
                      fill='black', 
                      width=30)
        x, y = event.x, event.y
        
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    image = cv.resize(img,(500,500), interpolation = cv.INTER_AREA)
    im = Image.fromarray(image)
    imagetk = ImageTk.PhotoImage(im, master=app)
    canvas.create_image(0,0, image=imagetk, anchor='nw')
    canvas.bind("<Button-1>", get_x_y)
    canvas.bind("<B1-Motion>", draw)
    
    HWND = canvas.winfo_id()
    print(HWND)

    while win32gui.IsWindow(HWND):
        rect = win32gui.GetWindowRect(HWND)
        #print(rect)
        imageSave = ImageGrab.grab()
        imageSave = imageSave.crop(rect)
        app.update()
        
    imageSave = np.array(imageSave)
    save = cv.cvtColor(imageSave,cv.COLOR_RGB2BGR)   
    save = cv.resize(save,(160,160),interpolation = cv.INTER_AREA)    
    return save
    

# p = cv.imread("test.jpg")

# cv.imshow("", p)
# cv.waitKey(0)
# cv.destroyAllWindows()


# p2 = occlude(p) 

# cv.imshow("", p2)
# cv.waitKey(0)
# cv.destroyAllWindows()
