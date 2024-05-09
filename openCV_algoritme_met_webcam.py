import cv2
import numpy as np

def detecteer_stukken_via_webcam():
    try:
        # Opstarten van de webcam
        cap = cv2.VideoCapture(0)  # De parameter 0 geeft aan dat we de eerste webcam gebruiken

        # Controleren of de webcam correct is geopend
        if not cap.isOpened():
            print("Kan de webcam niet openen. Controleer of deze is aangesloten en probeer het opnieuw.")
            return None

        # Lees een frame van de webcam
        ret, frame = cap.read()

        # Controleren of het frame correct is gelezen
        if not ret:
            print("Kan geen frame van de webcam lezen.")
            cap.release()
            return None

        # Sluit de webcam
        cap.release()

        # Definieer de kleurwaarden voor rood en blauw
        ondergrens_rood = np.array([0, 0, 100])
        bovengrens_rood = np.array([100, 100, 255])
        ondergrens_blauw = np.array([100, 0, 0])
        bovengrens_blauw = np.array([255, 100, 100])

        # Converteer het frame naar HSV-kleurschema
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Binarizeer het frame om alleen de rode en blauwe stukken te isoleren
        masker_rood = cv2.inRange(hsv_frame, ondergrens_rood, bovengrens_rood)
        masker_blauw = cv2.inRange(hsv_frame, ondergrens_blauw, bovengrens_blauw)

        # Combineer de rode en blauwe maskers
        gecombineerd_masker = cv2.bitwise_or(masker_rood, masker_blauw)

        # Breng morfologische operaties aan om ruis te verminderen
        gecombineerd_masker = cv2.morphologyEx(gecombineerd_masker, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        # Vind contouren in het gecombineerde masker
        contouren, _ = cv2.findContours(gecombineerd_masker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lijst om de gedetecteerde kleuren en coordinaten op te slaan
        gedetecteerde_stukken = []

        # Loop door alle contouren en bepaal de kleur en coordinaten van elk stuk
        for contour in contouren:
            oppervlakte = cv2.contourArea(contour)
            if oppervlakte > 500:  # Drempelwaarde Hoe groot moet het zijn om mee te tellen?
                x, y, w, h = cv2.boundingRect(contour)

                # Bereken het middelpunt van de begrensde rechthoek
                middelpunt_x = x + w // 2
                middelpunt_y = y + h // 2

                # Bepaal de gemiddelde kleur in het stuk
                stuk = frame[y:y+h, x:x+w]
                gemiddelde_kleur = np.mean(stuk, axis=(0, 1))

                # Bepaal of het stuk rood of blauw is op basis van de gemiddelde kleur
                if gemiddelde_kleur[0] > gemiddelde_kleur[2]:  # Als de gemiddelde waarde van blauw groter is dan rood
                    kleur = "Blauw"
                else:
                    kleur = "Rood"

                # Voeg de kleur en het middelpunt van het stuk toe aan de lijst
                gedetecteerde_stukken.append((kleur, (middelpunt_x, middelpunt_y)))

        return gedetecteerde_stukken

    except Exception as e:
        print(f"Fout bij het vastleggen van frame van de webcam: {e}")
        return None
    finally:
        if gedetecteerde_stukken:
            print("Gedetecteerde stukken en coordinaten:")
            for kleur, (x, y) in gedetecteerde_stukken:
                print(f"Kleur: {kleur}, Coordinaten: ({x}, {y})")
    

detecteer_stukken_via_webcam()