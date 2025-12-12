<span style="color:#0b5394; font-size:38px;">📘 Compte Rendu Complet : YOLOv8 + Régression</span>
<span style="color:#38761d; font-size:30px;">1. 🌟 Introduction</span>

Ce projet combine :

<span style="color:#1155cc;"><b>YOLOv8</b></span> → détection de joueurs, ballon et objets

<span style="color:#cc0000;"><b>Régression</b></span> → analyse et prédiction (distance, vitesse, position)

L'idée : YOLO détecte, la régression explique et prédit.

<span style="color:#674ea7; font-size:30px;">2. 📦 Importation des Bibliothèques</span>
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

<span style="color:#990000; font-size:30px;">3. 🤖 Chargement de YOLOv8</span>
model = YOLO("yolov8n.pt")


<b>yolov8n.pt</b> : version la plus rapide

idéale pour la détection en temps réel

<span style="color:#6aa84f; font-size:30px;">4. 📂 Données (data.yaml)</span>
yaml_path = "/kaggle/input/data-updated/data.yaml"


Contient :

chemins d’images

annotations

classes : player, ball, referee…

<span style="color:#0c343d; font-size:30px;">5. 🏋️ Entraînement du Modèle</span>
model.train(data=yaml_path, epochs=50, imgsz=640)


<span style="color:#38761d;">✔ Ajuste YOLO pour reconnaître les objets du football</span>

<span style="color:#741b47; font-size:30px;">6. 📊 Évaluation</span>
metrics = model.val()


YOLO calcule :

précision

recall

mAP

<span style="color:#134f5c; font-size:30px;">7. 🔍 Détection Image</span>
results = model("image.jpg")
results[0].show()


Affiche :

boîtes

labels

scores

<span style="color:#3d85c6; font-size:30px;">8. 🎥 Détection Vidéo</span>
model.predict(source="video.mp4", show=True)


Détection image-par-image en temps réel.

<span style="color:#cc0000; font-size:38px;">9. Pourquoi utiliser la Régression ?</span>

YOLO → <span style="color:#0b5394;">détecte</span>,
Régression → <span style="color:#38761d;">explique + prédit</span>

4 raisons importantes :
🔹 <span style="color:#1155cc;">1. Comprendre les relations</span>

Exemple : vitesse ↣ distance au ballon ?

🔹 <span style="color:#cc4125;">2. Faire des prédictions</span>

Position future, vitesse future, proximité du ballon.

🔹 <span style="color:#6aa84f;">3. Donner un sens aux données YOLO</span>

YOLO donne des nombres →
La régression explique pourquoi ils changent.

🔹 <span style="color:#674ea7;">4. Analyse tactique</span>

Comportements, déplacements, influence des actions.

<span style="color:#f1c232; font-size:36px;">10. Analyse de Régression</span>
<span style="color:#134f5c; font-size:28px;">10.1 Régression Linéaire</span>
from sklearn.linear_model import LinearRegression

X = np.array(df["player_speed"]).reshape(-1,1)
y = df["distance_to_ball"]

model_reg = LinearRegression()
model_reg.fit(X, y)

Interprétation :

coef_ positif → plus il va vite, plus il s’éloigne

coef_ négatif → plus il va vite, plus il se rapproche

<span style="color:#38761d; font-size:28px;">10.2 Régression Polynomiale</span>
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)


Utile pour les relations non linéaires (courbes).

<span style="color:#0b5394; font-size:28px;">10.3 Visualisation</span>
plt.scatter(X, y)
plt.plot(X, model_reg.predict(X), linewidth=3)

<span style="color:#741b47; font-size:38px;">11. YOLOv8 + Régression = Analyse Complète</span>
<table style="width:100%; font-size:20px;"> <tr> <td style="color:#0b5394;"><b>YOLOv8</b></td> <td style="color:#38761d;"><b>Régression</b></td> </tr> <tr> <td>Détecte</td> <td>Explique</td> </tr> <tr> <td>Donne positions</td> <td>Donne relations</td> </tr> <tr> <td>Produits bruts</td> <td>Prédictions</td> </tr> <tr> <td>Vision</td> <td>Analyse</td> </tr> </table>
<span style="color:#990000; font-size:38px;">12. Conclusion</span>

Grâce à ce projet :

YOLOv8 détecte automatiquement joueurs + ballon

La régression analyse leurs mouvements

Ensemble → un outil puissant pour l’analyse sportive

Résultat : vision + intelligence + prédiction
