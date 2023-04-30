import PySimpleGUI as sg
import os
import cv2 as cv
from PIL import Image, ImageTk
import cropper
import occluder
import compare
from datetime import datetime

def cv2tk(img):
    timg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    tim = Image.fromarray(timg)
    tk_img = ImageTk.PhotoImage(image=tim)
    return tk_img

image = None

sg.theme("tan")
main_layout = [[sg.Button("Load Image")],[sg.Button("Crop")],[sg.Button("Use Webcam")],[sg.Button("Display")],[sg.Button("Occlude")],[sg.Button("Compare")],[sg.Button("Generate Embeddings")],[sg.Button("EXIT")]]
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
                    image = cv.imread(filename)
                    img_window["-IMAGE-"].update(data=cv2tk(image))
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
            
            img = cv2tk(frame)
            wc_window["-IMAGE-"].update(data=img)
            
            if wc_event == "Take Photo":
                image = cropper.skin_cropper(frame)
                w_name = "Cropped Image"
                cv.namedWindow(w_name)
                cv.moveWindow(w_name, 1100,150)
                cv.imshow(w_name, image)
              
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
    if main_event == "Compare":
        if image is not None:
            pred = compare.compare(image)
            if pred is not None:
                pred_img = cv.imread(pred)
                cmp_layout = [
                    [
                     sg.Column(
                         [
                             [
                                 sg.Text("Input Image")
                                 ],
                             [
                                 sg.Image(key="-IMAGE-"),  
                                 ],
                             ]
                         ),
                    sg.VSeperator(),
                    sg.Column(
                        [
                            [
                                sg.Text("Match")
                                ],
                            [
                                sg.Image(key="-CMP-"),  
                                ],
                            ]
                        ),
                    ]
                ]
                cmp_window = sg.Window("Prediction", cmp_layout)
                load = False
                while True:
                    cmp_event, cmp_values = cmp_window.read(timeout=1)
                    if load == False:
                        cmp_window["-IMAGE-"].update(data=cv2tk(image))
                        cmp_window["-CMP-"].update(data=cv2tk(pred_img))
                        load = True
                    
                    if cmp_event == sg.WIN_CLOSED:
                        break

                cmp_window["-IMAGE-"].update(data=None)
                cmp_window["-CMP-"].update(data=None)        
                cmp_window.close()
            else:
                add_layout = [
                    [sg.Text("This person does not appear to be in the database would you like to add them?")],
                    [sg.Button("Yes")],
                    [sg.Button("No")],
                    ]
                add_window = sg.Window("Add to Database?", add_layout)
                while True:
                    add_event,add_values = add_window.read()
                    if add_event == "Yes":
                        folder_layout = [
                            [sg.Text('Enter Name for Database:')],
                            [sg.Input('', enable_events=True, key='-INPUT-', font=('Arial Bold', 20), expand_x=True, justification='left')],
                            [sg.Button('Ok')],
                            ]
                        folder_window = sg.Window("Name Input", folder_layout)
                        while True:
                            fold_event, fold_values = folder_window.read()
                            if fold_event == 'Ok' or fold_event == sg.WIN_CLOSED:
                                folder = str(fold_values['-INPUT-'])
                                break
                        folder_window.close()
                        
                        folder_path = "data/database/" + folder 
                        print(folder_path)
                        os.mkdir(folder_path)
                        folder_path = folder_path + "/"
                        cv.imwrite(folder_path + datetime.now().strftime("%H%M%S.jpg"),image)
                        break
                    if add_event == "No" or add_event == sg.WIN_CLOSED:
                        break
                
                add_window.close()
        else:
            error_layout = [[sg.Text("No Image Selected")]]
            error_window = sg.Window(title = "Error", layout = error_layout).read()
    
    if main_event == "Generate Embeddings":        
        with open("get_embeddings.py") as f:
            exec(f.read())
        succ_layout = [[sg.Text("Embeddings Generated")]]
        succ_window = sg.Window(title = "Success", layout = succ_layout).read()
            
    if main_event == "EXIT" or main_event == sg.WIN_CLOSED:
        break

main_window.close()