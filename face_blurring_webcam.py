import cv2
# Détection des visages
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# VideoCapture C'est une fonction,Pour capturer des vidéos d'une caméra connectée au système
# Tu peux passer. 0 Ou 1
# 0 Pour les webcams portables
# 1 Pour webcam externe
video_capture = cv2.VideoCapture(0)
# UnwhileCycles infinis,Capturez un nombre illimité d'images pour la vidéo,Parce que la vidéo est une combinaison de cadres
while True:
    # Capture des dernières images de la vidéo
    check, frame = video_capture.read()
    # Convertir le cadre en niveaux de gris（Des ombres noires et blanches）
     
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection de plusieurs visages dans un cadre capturé
    # scaleFactor：Le paramètre indique la réduction de la taille de l'image à chaque échelle d'image.
    # minNeighbors: Le paramètre indique combien de voisins Chaque rectangle doit avoir pour le garder.
    # Le rectangle contient l'objet d'essai.
    # L'objet ici est le visage humain.
    face = cascade.detectMultiScale(gray_image, scaleFactor=2.0, minNeighbors=4)
    for x, y, w, h in face:
        # Dessiner une bordure autour du visage détecté.
        #（La couleur de la bordure ici est verte,L'épaisseur est3）
        image = cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 3)
        # Masquer les visages dans un rectangle
        image[y:y+h, x:x+w] = cv2.medianBlur(image[y:y+h, x:x+w],35)

        # Afficher des visages flous dans la vidéo
        cv2.imshow('face blurred', frame)
        key = cv2.waitKey(1)
        # L'instruction ne fonctionne qu'une seule fois par image.
        # En gros,,Si on avait une clé,Et cette clé est une q
        if key == ord('q'):
            break


# Nous suspendrons la sortie while Cycle,
# Et courir：
video_capture.release()
cv2.destroyAllWindows()
