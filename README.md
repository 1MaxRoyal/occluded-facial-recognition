# occluded_face_recognition

facial recognition of occluded faces

trained on the pretrained vggface model from Tim Esler's [facenet-pytorch repo](https://github.com/timesler/facenet-pytorch)

the [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html) has been used to retrain the model from facenet-pytorch to recognise occluded faces. each subject from the database has been given a random word as a classifier
thus the apparent randomness of naming in any log files