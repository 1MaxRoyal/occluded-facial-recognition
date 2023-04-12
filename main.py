import PySimpleGUI as sg
import os
import cv2 as cv
from PIL import Image, ImageTk
import cropper
import occluder
import predicter

image = None

sg.theme("tan")
main_layout = [[sg.Button("Load Image")],[sg.Button("Use Webcam")],[sg.Button("Display")],[sg.Button("Crop")],[sg.Button("Occlude")],[sg.Button("Predict")],[sg.Button("EXIT")]]
main_window = sg.Window(title="Facial Recognition", layout=main_layout, margins=(100, 50), location=(100,100))

# event loop
while True:
    main_event, main_values = main_window.read()
    
    if main_event == "Load Image":
        img_layout = [
            [
            sg.Column(
                [
                    [
                        sg.Text("Image Folder"),
                        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
                        sg.FolderBrowse(),
                        ],
                    [
                        sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")
                        ],
                    [
                        sg.Button("Select Image")
                        ],
                    [
                        sg.Button("EXIT")
                        ],
                    ]), 
            sg.VSeperator(),
            sg.Column(
                [
                    [
                        sg.Text("Use 'Browse' to select image folder.")
                        ],
                    [
                        sg.Text("You must use the 'Select Image' button to save image choice.")
                        ],
                    [
                        sg.Image(key="-IMAGE-"),  
                        ],
                    ]
                ),
            ]
        ]
        
        img_window = sg.Window(title=("Load Image"), layout=img_layout,margins=(100, 50), location=(150,100))
        
        while True:
            img_event, img_values = img_window.read()
            
            if img_event == "EXIT" or img_event == sg.WIN_CLOSED:
                image = None
                break
            
            if img_event == "Select Image":
                break
            
            if img_event == "-FOLDER-":
                folder = img_values["-FOLDER-"]
                try:
                    # Get list of files in folder
                    file_list = os.listdir(folder)
                except:
                    file_list = []
            
                fnames = [
                    f
                    for f in file_list
                    if os.path.isfile(os.path.join(folder, f))
                    and f.lower().endswith((".png", ".jpg"))
                ]
                img_window["-FILE LIST-"].update(fnames)
                
            elif img_event == "-FILE LIST-":  # A file was chosen from the listbox
                try:
                    filename = os.path.join(
                        img_values["-FOLDER-"], img_values["-FILE LIST-"][0]
                    )
                    print(filename)
                    image = cv.imread(filename,1)
                    img = cv.cvtColor(image,cv.COLOR_BGR2RGB)
                    im = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=im)
                    img_window["-IMAGE-"].update(data=imgtk)
                except:
                    pass
                
        img_window.close()
        
    if main_event == "Use Webcam":
        wc_layout = [
            [
                sg.Image(key = "-IMAGE-")
                ],
            [
                sg.Button("Take Photo")
                ],   
            [
                sg.Button("Exit")
                ], 
            ]
        
        wc_window = sg.Window(title = "Webcam", layout=wc_layout,margins=(100, 50), location=(150,100) )
        wc = cv.VideoCapture(0) 
        
        while True:
            wc_event, wc_values = wc_window.read(timeout=1)
            
            if wc_event == sg.WIN_CLOSED or wc_event == "Exit":
                wc_window["-IMAGE-"].update(data=None)
                break
            
            
            _, frame = wc.read()
            frame = cv.flip(frame,1)
            
            img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im)
            wc_window["-IMAGE-"].update(data=imgtk)
            
            if wc_event == "Take Photo":
                image = frame
                cv.imshow("Taken Image", image)
              
        wc.release()
        wc_window.close()
        cv.destroyAllWindows()
            
    
    if main_event == "Display" and image is not None:
       cv.imshow("Selected Image", image)
       cv.waitKey(0)
       cv.destroyAllWindows()
       
    if main_event == "Crop":
        if image is not None:
            image = cropper.skin_cropper(image)
            cv.imshow("Cropped Image", image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            error_layout = [[sg.Text("No Image Selected")]]
            error_window = sg.Window(title = "Error", layout = error_layout).read()
            
    if main_event == "Occlude":
        if image is not None:
            image = occluder.occlude(image)
                
        else:
            error_layout = [[sg.Text("No Image Selected")]]
            error_window = sg.Window(title = "Error", layout = error_layout).read()
        
    if main_event == "Predict":
        if image is not None:
            pred = predicter.predict(image)
        else:
            error_layout = [[sg.Text("No Image Selected")]]
            error_window = sg.Window(title = "Error", layout = error_layout).read()
        
    if main_event == "EXIT" or main_event == sg.WIN_CLOSED:
        break

main_window.close()