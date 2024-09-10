'''Ce fichier contient les fonctions utiles et générique du Campus Manager. Elle contient entre autres des fonctions pour des formattages de dates.'''

from datetime import datetime, timezone, timedelta
import json

def change_to_datetime(date_string):
    '''
    Change le string de date dans un format datetime. Exemple de date_string  : "2024-01-26-12-35-12-10"
    %Y représente l'année à quatre chiffres, %m le mois, %d le jour,
    %H l'heure, %M les minutes, %S les secondes et %f les microsecondes (vous devez multiplier les millisecondes par 1000 pour obtenir des microsecondes)
    '''
    return datetime.fromisoformat(date_string)


def order_date(list_date,index):
    '''
    Mets les dates d'une liste de datetime dans l'ordre de la plus ancienne date à la plus récente.
    '''
    sorted_data = sorted(list_date, key=lambda x: x[index])
    return sorted_data


def actual_datetime():
    '''
    Renvoie la date et l'heure actuelle sous format datetime.
    '''
    return datetime.now(timezone.utc)


def shorten_datetime(date):
    '''
    Réduit le datetime pour ne pas prendre en compte les heures, minutes et secondes (heure par défaut : minuit).
    '''
    return date.replace(hour= 0, minute = 0,second=0)


def modify_json(json_path, new_values):
    '''
    Modifie le fichier json à deux profondeurs.
    '''

    # Lire le fichier JSON
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Mettre à jour les valeurs du JSON
    for key1, value1 in new_values.items():
        if key1 in data:
            for key2, value2 in value1.items():
                if key2 in data[key1]:
                    data[key1][key2] = value2
    
    # Écrire les changements dans le fichier JSON
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)