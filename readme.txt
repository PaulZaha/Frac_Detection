Schritt 1: Image_preaugmentation.py
Skript ausführen, comments stehen in Skript.

Schritt 2: Dataframe_preaugmentation.py
Zwingend nach Schritt 1 auszuführen, da genau die Bilder aus dem Dataframe geschmissen werden, die in Schritt 1
aus dem directory entfernt werden.

Schritt 3: pipelines.py
Aktuell auf das EfficientNetB4 zugeschnitten (mit targetsize=380x380 in main-methode)

Erklärung main in pipelines.py:
randomized splitting in train und test dataset
erstellen der generators
mit z.B. "EfficientNetB4(train_generator,validation_generator,test_generator,gewicht)" (Zeile 184)
kann modell traininiert und evaluiert werden.
Hier zugriff auf models.py

In models.py wird Methode des jeweiligen Modells aufgerufen
Alle Callbacks sowie compiling, fitting und evaluation werden in der Modell-Methode gecallt

