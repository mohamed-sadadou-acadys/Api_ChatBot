'''Ce fichier contient la configuration de l'application FLASK du Campus Manager, dont les fonctions d'appels API disponibles.'''

from flask import Flask, request, jsonify, got_request_exception
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import logging
from urllib.parse import unquote
import numpy as np
from API.database import Database
from RAG.scripts.Embeddings import WordEmbedding
from RAG.scripts.ExternalEmbeddings import ExternalResourcesEmbedding
from RAG.scripts.Generation import RAG
from API.app_func import setup_model, thread_chatbot, send_wait, initialize_scheduler, delete_conversations_by_date, preprocess_new_data, merge_n_token, update_scheduler, server_gpu, preprocess_external_data
from API.utils import modify_json
import json

### CONFIGURATION DU SERVEUR ###

app = Flask(__name__)
root = ""
socketio = SocketIO(app,cors_allowed_origins="*",ping_timeout=120, ping_interval=25,
                    logger=True, engineio_logger=True,supports_credentials=True,transports=['polling'])

# Gestion des informations provenant de nginx
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Gestion des CORS
CORS(app, 
     resources={r"/*": {"origins": "*", 
                        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
                        "allow_headers": ["Authorization", "Content-Type"]}
                })

# Gestion des logs
def log_request(sender, exception, **extra):
    logging.info(f"Headers: {request.headers}")
got_request_exception.connect(log_request, app) # Affichage des headers lors du print des logs

logging.basicConfig(level=logging.DEBUG) # Affichage de toutes les alertes de niveau DEBUG 
app.logger.setLevel(logging.DEBUG)

@socketio.on_error()  # Pour capturer toutes les erreurs socketio 
def error_handler(e):
    app.logger.error(f'Erreur Socket.IO: {str(e)}')

@socketio.on_error_default  # Pour capturer toutes les erreurs non gérées
def default_error_handler(e):
    app.logger.error(f'Erreur par défaut Socket.IO: {str(e)}')

@app.before_request # Pour afficher les requêtes entrantes
def before_request_logging():
    app.logger.debug(f"Requête entrante: {request.method} {request.url}")


### INITIALISATION DES VARIABLES ###

# Ouvre le fichier json avec les variables de configuration
config_path = "RAG/config/config.json"
with open(config_path, 'r') as config_file:
        var = json.load(config_file)
        var_server = var.get("server")
        var_RAG = var.get("RAG")
        var_scheduler = var.get("scheduler")

with app.app_context():

    app.config['gpu'] = server_gpu(var_server.get("gpu"))
    app.config['Database'] = Database() 
    app.config['WordEmbedding'] = WordEmbedding()
    app.config['ExternalResourcesEmbedding'] = ExternalResourcesEmbedding()
    app.config['RAG'] = RAG(var_RAG)
    app.config['queueRequest']= list() 
    app.config['Scheduler'] = BackgroundScheduler()

    # Initialisation des modèles sur chaque GPU sélectionnée
    for n in app.config['gpu'].get_gpu() :
        app.config[f'gpu:{n}'] = setup_model(app.config['RAG'],n)

    # Initilisation du scheduler de suppression des conversations
    initialize_scheduler(app,app.config['Scheduler'],var_scheduler.get("day_period"),var_scheduler.get("hour"),var_scheduler.get("minute"))

    # Lancement du chatbot en arrière plan (en attente tant qu'il n'y a pas de requête)
    threading.Thread(target=thread_chatbot, args=(app, socketio)).start()

    # Lancement de la communication avec la liste d'attente (envoie wait toutes les 25sec)
    socketio.start_background_task(send_wait, app, socketio)




### CONNECTION ###

@socketio.on('connect')
def handle_connect():
    '''
    Effectue la connection entre un utilisateur et l'application
    '''
    print('Un client est connecté')


@socketio.on('listener')
def event_happened(data):
    '''
    Gère les events wait et stop provenant d'un utilisateur 
    '''

    # Affiche l'event wait
    if data['message'] == "wait":
        print(f'Wait received from {request.sid}')


    # Arrête les processus de génération/liste d'attente lors de l'event stop
    elif data['message'] == "stop" : 
        print('SSSSSSSTTTTTTTTTTTTTOOOOOOOOOOOOOOPPPPPPPPPPPPPPP')
        

        for n in app.config['gpu'].get_gpu():
            if request.sid == app.config[f'gpu:{n}']['currentSid'] :
                # Active le critère d'arrêt du LLM pour la génération en cours 
                app.config[f'gpu:{n}']['stopping_criteria'].change_to_True()
        
        for i,requestInQueue in enumerate(app.config['queueRequest']):
            if request.sid == requestInQueue['sid']:
                print(requestInQueue['sid'])
                # Supprime la requête de la liste d'attente s'il n'est pas en train de générer
                del app.config['queueRequest'][i]

        socketio.emit('event', {'message':'stop'},room= request.sid)



### CONVERSATION ###    

@socketio.on('create_chat')
def create_chat_stream(json_file=None):
    '''
    Réceptionne le premier chat d'une conversation et mise en place du stream
    '''

    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "question": {"type": "string"}
    },
    "required": ["id_user","question"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        socketio.emit('event', {'message': str(e)})
        return 

    # Ajout de la requête create_chat à la liste d'attente
    app.config['queueRequest'].append({
                                        "sid":request.sid,
                                        "id_user" : json_file["id_user"],
                                        "question" : json_file["question"],
                                        "event" : 'create_chat',
                                        "id_conv" : "0"
                                    })  
    

    print('longueur queue request : ',len(app.config['queueRequest']))



@socketio.on('add_chat')
def add_chat_stream(json_file=None):
    '''
    Receptionne un chat d'une conversation existante et mise en place du stream
    '''

    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "question": {"type": "string"}
    },
    "required": ["id_user","id_conv","question"]
    }
    
    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        socketio.emit('event', {'message': str(e)})
        return 

    # Ajout de la requête add_chat à la liste d'attente
    app.config['queueRequest'].append({
                                        "sid":request.sid,
                                        "id_user" : json_file["id_user"],
                                        "question" : json_file["question"],
                                        "event" : 'add_chat',
                                        "id_conv" : json_file["id_conv"]
                                    })
    

    print('longueur queue request : ',len(app.config['queueRequest']))
    print('queue generation empty : ', [app.config[f'gpu:{n}']['generator'].queue.empty() for n in app.config["gpu"].get_gpu()],
          )



@app.route(f'{root}/chat', methods=['GET'])
def get_conversation():
    '''
    Affiche la conversation d'un utilisateur en entier ou par page
    '''
        
    json_file = {}
    json_file['id_user'] = request.args.get('id_user')
    json_file['id_conv'] = request.args.get('id_conv')

    if 'i_start' in request.args and 'i_end' in request.args :
        # Récupération des indices de message start et end d'une conversation s'ils existent
        json_file['i_start'] = request.args.get('i_start')
        json_file['i_end'] = request.args.get('i_end')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "i_start" : {"type": "string",
                    "pattern": "^-?[0-9]+$"},
        "i_end" : {"type": "string",
                    "pattern": "^-?[0-9]+$"},
    },

    "required": ["id_user","id_conv"]
    }
    
    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    if 'i_start' in request.args and 'i_end' in request.args :

        # Convertis les indices en entier 
        i_start = int(request.args.get('i_start'))
        i_end = int(request.args.get('i_end'))

    else : 
        i_start = 0
        i_end = 5

    # Récupère la conversation recherchée
    conversation = app.config['Database'].getOneConversation(json_file['id_user'],
                                                               json_file['id_conv'])

    if conversation is None :
        return {'error': 'id_user or id_conv not in database'}, 404
    

    len_exchanges = len(conversation['Conversations'][0]['Exchanges'])

    # Récupère les messages entre les indices de début et de fin recherchés ou réels
    exchanges = conversation['Conversations'][0]['Exchanges'][min(i_start,len_exchanges):min(i_end,len_exchanges)]

    return jsonify({'Exchange':{i_start+i:(exchanges[i]['answer'],
                                           exchanges[i]['question']) for i in range(len(exchanges))},
                    'len_exchanges':len_exchanges,
                    'notation':conversation['Conversations'][0]['notation'],
                    'comment':conversation['Conversations'][0]['comment'],
                    'title':conversation['Conversations'][0]['title']}), 200



@app.route(f'{root}/chat', methods=['DELETE'])
def delete_conversation():
    '''
    Supprime une ou plusieurs conversations
    '''

    json_file = request.json
    json_file['id_user'] = request.args.get('id_user')

    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "ids_conv": {"type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^[0-9a-fA-F]{24}$"
                        }
                    }
        },
    "required": ["id_user","ids_conv"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    # Récupère l'ensemble des conversations
    conversations_dict = app.config['Database'].getAllConversations(json_file["id_user"])

    if conversations_dict is None :
        return {'error': 'id_user not in database'}, 404
    
    # Transforme l'ID de la conversation en indice de la variable conversations
    is_to_delete = app.config['Database'].changeIDconvToI(conversations_dict["Conversations"],
                                                             json_file["ids_conv"])

    # Supprime les conversations
    n_deleted = app.config['Database'].deleteConversation(json_file["id_user"],
                                                  is_to_delete,
                                                  conversations_dict["Conversations"])

    return jsonify({'element_deleted': n_deleted}), 200



### TITLE ###

@app.route(f'{root}/chat/title', methods=['GET'])
def get_titles():
    '''
    Récupère les titres de conversations d'un utilisateur en entier ou par page
    '''

    json_file = {}
    json_file['id_user'] = request.args.get('id_user')

    if 'i_start' in request.args  and 'i_end' in request.args :

        json_file['i_start'] = request.args.get('i_start')
        json_file['i_end'] = request.args.get('i_end')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "i_start" : {"type": "string",
                    "pattern": "^-?[0-9]+$"},
        "i_end" : {"type": "string",
                    "pattern": "^-?[0-9]+$"},
    },

    "required": ["id_user"]
    }
    
    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400


    if 'i_start' in request.args and 'i_end' in request.args :

        i_start = int(request.args.get('i_start'))
        i_end = int(request.args.get('i_end'))

    else : 
        i_start = 0
        i_end = 10
    

    # Récupération des conversations d'un utilisateur entre les indices de start et de end
    conversations = app.config['Database'].getManyConversations(json_file["id_user"],
                                                                i_start,
                                                                i_end)
    
    if conversations is None :
        return {'error': 'id_user not in database'}, 404

    if conversations['nb_conv'] == 0 : # Il n'y a pas de conversations
        return jsonify({'nb_conv':0,'titles':{}}), 200

    if conversations['Conversations'] == [] :
        return {'error': 'i_start to i_end not in database'}, 404

    return jsonify({'nb_conv':conversations["nb_conv"],
                    'titles':{i_start+i:[str(conversations["Conversations"][i]["_id"]),
                                          conversations["Conversations"][i]['title']] for i in range(len(conversations["Conversations"]))}}), 200


@app.route(f'{root}/chat/title', methods=['PUT'])
def update_title():
    '''
    Mets à jour le titre d'une conversation d'un utilisateur en entier ou par page
    '''

    json_file = request.json
    json_file['id_user'] = request.args.get('id_user')
    json_file['id_conv'] = request.args.get('id_conv')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "title" : {"type": "string"},
    },

    "required": ["id_user","id_conv","title"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Mets à jour le titre
    matched_count = app.config['Database'].updateObjectsInConv(["title"],
                                                json_file['id_user'],
                                                json_file['id_conv'],
                                                [json_file['title']])
    
    if matched_count == 0 :
        return {'error': 'id_user or id_conv not in database'}, 404

    return jsonify({}), 204


### NOTATION ###

@app.route(f'{root}/chat/notation', methods=['GET'])
def get_notation():
    '''
    Récupère la note d'une conversations d'un utilisateur
    '''

    json_file = {}
    json_file['id_user'] = request.args.get('id_user')
    json_file['id_conv'] = request.args.get('id_conv')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"}
    },

    "required": ["id_user","id_conv"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    # Récupère la conversation recherchée
    conversation = app.config['Database'].getOneConversation(json_file['id_user'],
                                                            json_file['id_conv'])

    if conversation is None :
        return {'error': 'id_user or id_conv not in database'}, 404
    
    
    return jsonify({'notation':conversation['Conversations'][0]['notation'],
                    'comment':conversation['Conversations'][0]['comment']}), 200



@app.route(f'{root}/chat/notation', methods=['PUT'])
def update_notation():
    '''
    Mets à jour une note d'un utilisateur
    '''

    json_file = request.json
    json_file['id_user'] = request.args.get('id_user')
    json_file['id_conv'] = request.args.get('id_conv')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "notation" : {"type": "integer"},
        "comment" : {"type": "string"},
    },

    "required": ["id_user","id_conv","notation","comment"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Mets à jour la note
    matched_count = app.config['Database'].updateObjectsInConv(['notation','comment'],
                                                    json_file['id_user'],
                                                    json_file['id_conv'],
                                                    [json_file['notation'],json_file['comment']])
    
    if matched_count == 0 :
        return {'error': 'id_user or id_conv not in database'}, 404
    
    # Récupère les messages de la conversation
    exchanges = app.config['Database'].getOneConversation(json_file['id_user'],
                                                               json_file['id_conv'])['Conversations'][0]['Exchanges']

    if len(exchanges)==0:
        return jsonify({}), 200

    # Renvoie la dernière question et la dernière réponse de la conversation
    return jsonify({'Exchange':[exchanges[0]['answer'],exchanges[0]['question']]}), 200

        


@app.route(f'{root}/chat/notation', methods=['DELETE'])
def delete_notation():
    '''
    Supprime une note d'un utilisateur
    '''

    json_file = {}
    json_file['id_user'] = request.args.get('id_user')
    json_file['id_conv'] = request.args.get('id_conv')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"}
    },

    "required": ["id_user","id_conv"]
    }

    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Mets à jour la note pour qu'elle apparaisse vide
    matched_count = app.config['Database'].updateObjectsInConv(['notation','comment'],
                                                    json_file['id_user'],
                                                    json_file['id_conv'],
                                                    [0,''])

    if matched_count == 0 :
        return {'error': 'id_user or id_conv not in database'}, 404

    return jsonify({}), 204


### REMARK ###

@app.route(f'{root}/chat/remark', methods=['GET'])
def get_remark():
    '''
    Récupère la remarque d'un expert
    '''

    json_file = {}
    json_file['id_user'] = request.args.get('id_user')
    json_file['id_conv'] = request.args.get('id_conv')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"}
    },

    "required": ["id_user","id_conv"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    # Récupère la conversation recherchée
    conversation = app.config['Database'].getOneConversation(json_file['id_user'],
                                                            json_file['id_conv'])

    if conversation is None :
        return {'error': 'id_user or id_conv not in database'}, 404
    
    
    return jsonify({'remark':conversation['Conversations'][0]['remark']}), 200


@app.route(f'{root}/chat/remark', methods=['PUT'])
def update_remark():
    '''
    Mets à jour la remarque d'un expert
    '''

    json_file = request.json
    json_file['id_user'] = request.args.get('id_user')
    json_file['id_conv'] = request.args.get('id_conv')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "remark" : {"type": "string"},
    },

    "required": ["id_user","id_conv","remark"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    # Mets à jour la remarque d'un expert 
    matched_count= app.config['Database'].updateObjectsInConv(['remark'],
                                                            json_file['id_user'],
                                                            json_file['id_conv'],
                                                           [json_file['remark']])

    if matched_count == 0 :
        return {'error': 'id_user or id_conv not in database'}, 404

    return jsonify({}), 204


@app.route(f'{root}/chat/remark', methods=['DELETE'])
def delete_remark():
    '''
    Supprime la remarque d'un expert
    '''

    json_file = {}
    json_file['id_user'] = request.args.get('id_user')
    json_file['id_conv'] = request.args.get('id_conv')


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"}
    },

    "required": ["id_user","id_conv"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Mets à jour la remarque d'un expert pour qu'elle apparaisse vide
    matched_count = app.config['Database'].updateObjectsInConv(['remark'],
                                                            json_file['id_user'],
                                                            json_file['id_conv'],
                                                            [''])
    
    if matched_count == 0 :
        return {'error': 'id_user or id_conv not in database'}, 404

    return jsonify({}), 204


### DATA PREPROCESSING ###

@app.route(f'{root}/data', methods=['POST'])
def create_data_formation():
    '''
    Vectorise et place de nouveaux documents de formation dans le base de données RAG
    '''

    json_file = request.json
    json_file['id_formation'] = request.args.get('id_formation')


    schema = {
    "type": "object",
    "properties": {
        "id_formation": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "formation_title": {"type": "string"},
        "docs":{
            "type":"array",
            "items": {
                "type":"object",
                "properties":
                {
                "doc_title": {"type": "string"},
                "id": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
                "path": {"type": "string",
                          "pattern": r".*\.(ppt|pptx|doc|docx|pdf)$"}
                },
                "required": ["doc_title","id","path"]
            },
            "minItems": 1
        }
    },

    "required": ["id_formation","formation_title","docs"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    for doc in json_file['docs']:

        if app.config['WordEmbedding'].is_doc_in_db(doc['id']) :
            return jsonify({'error': f"id_doc='{doc['id']}' already in the database"}), 500


    for doc in json_file['docs']:
        # Prétraite le nouveau document à vectoriser
        df_formation = preprocess_new_data(doc['path'])

        # Vectorise le document dans la base de données 
        app.config['WordEmbedding'].add_documents(json_file['id_formation'],
                                                    doc['id'],
                                                    json_file['formation_title'],
                                                    doc['doc_title'],
                                                    df_formation)

        
    return jsonify({}), 204


@app.route(f'{root}/data', methods=['GET'])
def get_data_formation():
    '''
    Récupère les IDs de tous les documents de la base de données RAG
    '''

    json_file = {}

    if 'id_formation' in request.args :
        json_file['id_formation'] = request.args.get('id_formation')

    schema = {
    "type": "object",
    "properties": {
        "id_formation" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"}
    },

    "required": []
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Récuperation des ids des documents en fonction de l'id de formation s'il y en a un 
    documents_dict = app.config['WordEmbedding'].get_documents(json_file.get('id_formation'))

    return jsonify(documents_dict), 200



@app.route(f'{root}/data', methods=['PUT'])
def update_data_formation():
    '''
    Vectorise et met à jour des documents de formation RAG 
    '''

    json_file = request.json
    json_file['id_formation'] = request.args.get('id_formation')

    schema = {
    "type": "object",
    "properties": {
        "id_formation": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "formation_title": {"type": "string"},
        "docs":{
            "type":"array",
            "items": {
                "type":"object",
                "properties":
                {
                "doc_title": {"type": "string"},
                "id": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
                "path": {"type": "string",
                          "pattern": r".*\.(ppt|pptx|doc|docx|pdf)$"}
                },
                "required": ["doc_title","id","path"]
            },
            "minItems": 1
        }
    },

    "required": ["id_formation","formation_title","docs"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    for doc in json_file['docs']:

        if app.config['WordEmbedding'].is_doc_in_db(doc['id']) is None :
            return jsonify({"error": f"id_doc='{doc['id']}' doesn't exist in the database"}), 500
        

    for doc in json_file['docs']:
        # Prétraite le nouveau document à vectoriser
        df_formation = preprocess_new_data(doc['path'])

        # Vectorise le document dans la base de données en archivant l'ancien document
        app.config['WordEmbedding'].update_documents(json_file['id_formation'],
                                                        doc['id'],
                                                        json_file['formation_title'],
                                                        doc['doc_title'],
                                                        df_formation)

    return jsonify({}), 204


@app.route(f'{root}/data', methods=['DELETE'])
def delete_data_formation():
    '''
    Supprime des documents de la base de données RAG
    '''

    json_file = request.json

    schema = {
    "type": "object",
    "properties": {

        "docs":{
            "type":"array",
            "items": {
                "type":"object",
                "properties":
                {
                "id": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"}
                },
                "required": ["id"]
            },
            "minItems": 1
        }
    },

    "required": ["docs"]
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    
    for doc in json_file['docs']:

        if app.config['WordEmbedding'].is_doc_in_db(doc['id']) is None :
            return jsonify({"error": f"id_doc='{doc['id']}' doesn't exist in the database"}), 500
        

    for doc in json_file['docs']:
        # Supprime le document en fonction de son ID
        app.config['WordEmbedding'].delete_documents(doc['id'])

    return jsonify({}), 204


### DATA PREPROCESSING ###

@app.route(f'{root}/external_data', methods=['POST'])
def create_external_data():
    '''
    Vectorizes and places new external documents in the RAG database.
    '''

    json_file = request.json

    schema = {
        "type": "object",
        "properties": {
            "docs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_title": {"type": "string"},
                        "id": {
                            "type": "string",
                            "pattern": "^[0-9a-fA-F]{24}$"
                        },
                        "path": {
                            "type": "string",
                            "pattern": r".*\.(ppt|pptx|doc|docx|pdf)$"
                        }
                    },
                    "required": ["doc_title", "id", "path"]
                },
                "minItems": 1
            }
        },
        "required": ["docs"]
    }

    # Validate input data
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    for doc in json_file['docs']:
        if app.config['ExternalResourcesEmbedding'].is_doc_in_db(doc['id']):
            return jsonify({'error': f"id_doc='{doc['id']}' already in the database"}), 500

    for doc in json_file['docs']:
        # Preprocess the new document to vectorize
        df_data = preprocess_external_data(doc['path'])

        # Vectorize the document into the database
        app.config['ExternalResourcesEmbedding'].add_external_documents(
            id_doc=doc['id'],
            doc_title=doc['doc_title'],
            document_content=' '.join(df_data['Contenu'])
        )

    return jsonify({}), 204


@app.route(f'{root}/external_data', methods=['GET'])
def get_external_data():
    '''
    Retrieves the IDs of all documents in the RAG database.
    '''

    # Retrieve the list of document IDs in the database
    documents_list = app.config['ExternalResourcesEmbedding'].get_external_documents()

    return jsonify(documents_list), 200


@app.route(f'{root}/external_data', methods=['PUT'])
def update_external_data():
    '''
    Vectorizes and updates external documents in the RAG.
    '''

    json_file = request.json

    schema = {
        "type": "object",
        "properties": {
            "docs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_title": {"type": "string"},
                        "id": {
                            "type": "string",
                            "pattern": "^[0-9a-fA-F]{24}$"
                        },
                        "path": {
                            "type": "string",
                            "pattern": r".*\.(ppt|pptx|doc|docx|pdf)$"
                        }
                    },
                    "required": ["doc_title", "id", "path"]
                },
                "minItems": 1
            }
        },
        "required": ["docs"]
    }

    # Validate input data
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    for doc in json_file['docs']:
        if app.config['ExternalResourcesEmbedding'].is_doc_in_db(doc['id']) is None:
            return jsonify({"error": f"id_doc='{doc['id']}' doesn't exist in the database"}), 500

    for doc in json_file['docs']:
        # Preprocess the document to vectorize
        df_data = preprocess_external_data(doc['path'])

        # Vectorize the document into the database, archiving the old document
        app.config['ExternalResourcesEmbedding'].update_external_document(
            id_doc=doc['id'],
            doc_title=doc['doc_title'],
            document_content=' '.join(df_data['Contenu'])
        )

    return jsonify({}), 204


@app.route(f'{root}/external_data', methods=['DELETE'])
def delete_external_data():
    '''
    Deletes documents from the RAG database.
    '''

    json_file = request.json

    schema = {
        "type": "object",
        "properties": {
            "docs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "pattern": "^[0-9a-fA-F]{24}$"
                        }
                    },
                    "required": ["id"]
                },
                "minItems": 1
            }
        },
        "required": ["docs"]
    }

    # Validate input data
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    for doc in json_file['docs']:
        if app.config['ExternalResourcesEmbedding'].is_doc_in_db(doc['id']) is None:
            return jsonify({"error": f"id_doc='{doc['id']}' doesn't exist in the database"}), 500

    for doc in json_file['docs']:
        # Delete the document based on its ID
        app.config['ExternalResourcesEmbedding'].delete_external_document(doc['id'])

    return jsonify({}), 204


### DATA ANALYSIS ###

@app.route(f'{root}/analysis/token', methods=['GET'])
def get_token_analysis():
    '''
    Analyse les tokens traités et générés par conversation, utilisateur ou formation
    '''

    json_file = {}

    if 'id_user' in request.args :
        json_file['id_user'] = request.args.get('id_user')

    if 'id_conv' in request.args :
        json_file['id_conv'] = request.args.get('id_conv')

    if 'id_formation' in request.args :
        json_file['id_formation'] = request.args.get('id_formation')

    if 'date1' in request.args and 'date2' in request.args :
        json_file['date1'] = request.args.get('date1')
        json_file['date2'] = request.args.get('date2')
        
        # Pour transformer les URLs de date dans le bon format
        json_file['date1'] = unquote(json_file['date1'])
        json_file['date2'] = unquote(json_file['date2'])
        


    schema = {
    "type": "object",
    "properties": {
        "id_user": {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "id_conv" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"},
        "date1": {"type": "string",
                    "pattern": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}"},
        "date2": {"type": "string",
                    "pattern": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}"},
        "id_formation" : {"type": "string",
                    "pattern": "^[0-9a-fA-F]{24}$"}
    },

    "required": []
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Les types d'objet à récupérer de la base de données en fonction des éléments reçus : nombre de token et la date du dernier message
    object_to_find = ["n_token"] + (["last_update"] if json_file.get("date1") and json_file.get("date2") else [])
    

    if json_file.get('id_user') and json_file.get('id_conv'):

        # Récupération du nombre de token d'une conversation spécifique
        n_token_object = app.config['Database'].getObjectsInConv(json_file['id_user'],
                                                json_file['id_conv'],
                                                object_to_find)
        
        if n_token_object is None :
            return {'error': 'id_user or id_conv not in database'}, 404

        # Sommation des nombres de token en fonction des dates spécifiées
        n_token = merge_n_token([n_token_object],json_file.get("date1"),json_file.get("date2"))

    elif json_file.get('id_user'):

        # Récupération des nombres de token de toutes les conversation d'un utilisateur spécifique 
        n_token_object = app.config['Database'].getObjectsForUser(json_file['id_user'],
                                                    object_to_find)
            
        if n_token_object is None :
            return {'error': 'id_user not in database'}, 404

        # Sommation des nombres de token en fonction des dates spécifiées
        n_token = merge_n_token([n_token_object],json_file.get("date1"),json_file.get("date2"))

    else :
        
        # Récupération des nombres de token de toutes les conversations de tous les utilisateurs
        all_n_token_object = app.config['Database'].getObjects(object_to_find)

        if all_n_token_object is None :
            return {'error': 'There is no element in the database'}, 404

        # Sommation des nombres de token en fonction des dates spécifiées
        n_token = merge_n_token(all_n_token_object,json_file.get("date1"),json_file.get("date2"))


    if json_file.get('id_formation'):
        
        # Si l'ID de la formation voulu est bien présente, le nombre de token est affiché pour celle-ci uniquement
        if n_token.get(json_file.get('id_formation')):
            n_token = {json_file.get('id_formation'):n_token.get(json_file.get('id_formation'))}
        else :
            n_token = {json_file.get('id_formation'):0} 
    else :
        n_token

    return jsonify(n_token), 200


### PARAMETERS ###

@app.route(f'{root}/parameters', methods=['GET'])
def get_parameters():
    '''
    Récupère les paramètres du Campus Manager
    '''

    parameters = {"RAG": app.config['RAG'].get_dict_parameters()}

    # Les paramètres du scheduler de suppression de conversation sont contenus dans les jobs
    jobs = app.config['Scheduler'].get_jobs()

    parameters |= {"server" : {"gpu":app.config['gpu'].get_gpu()},
                   "scheduler": {"day_period":jobs[0].args[1],
                        "hour":jobs[0].args[2],
                        "minute":jobs[0].args[3]}}
        
    
    return jsonify(parameters), 200


@app.route(f'{root}/parameters', methods=['PUT'])
def update_parameters():
    '''
    Mets à jour les paramètres du Campus Manager
    '''

    json_file = request.json

    schema = {
    "type": "object",
    "properties": {
        "server" : {
                    "type":"object",
                    "properties":
                    {
                    "gpu": {"type":"array",
                            "items": {"type":"integer",
                                        "enum": [0, 1]},
                                    "uniqueItems": True,
                                    "minItems": 0,
                                    "maxItems": 2 },
                    },
                    "required": []
                },
                 
        "scheduler": {
                "type":"object",
                "properties":
                {
                    "hour": {"type": "integer",
                        "pattern": "^[0-2][0-9]$"},
                    "minute": {"type": "integer",
                        "pattern": "^[0-5][0-9]$"},
                    "day_period": {"type": "integer"}, 
                },
                "required": ["hour","minute","day_period"]
            },
        "RAG":{
                "type":"object",
                "properties":
                {
                    "model_path" : {"type": "string"},
                    "k_retriever" : {"type": "integer"},
                    "score_threshold_retriever": {"type": "number",
                                                    "minimum": 0,
                                                    "maximum": 1},
                    "temperature" : {"type": "number",
                                            "minimum": 0},
                    "top_p" : {"type": "number",
                                        "minimum": 0,
                                        "maximum": 1},
                    "top_k" : {"type": "integer",
                                        "maximum": 5000},
                    "max_length" : {"type": "integer"},
                    "max_new_tokens" : {"type": "integer"},
                    "repetition_penalty" : {"type": "number",
                                        "minimum": 0},
                    "ratio_start_index" : {"type": "number",
                                        "minimum": 0,
                                        "maximum": 1},
                    "decay_factor" : {"type": "number",
                                        "minimum": 0},
                    "stop_sequence_list" : {"type":"array",
                                            "items": {"type":"string"}},
                    "system_preprompt" : {"type": "string"},
                    "context_preprompt" : {"type": "string"},
                    "reformulate_query" : {"type": "boolean"}
            },
                "required": []
        },
    },

    "required": []
    }

    # Validation des données d'entrée
    try:
        validate(instance=json_file, schema=schema)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400


    if json_file.get("RAG",{}).get("model_path"):
        last_name = app.config['RAG'].model_path

        try :
            app.config['RAG'].model_path = json_file["RAG"]["model_path"]
            for n in app.config['gpu'].get_gpu() :
                # Libère la mémoire GPU des anciens modèles 
                app.config['RAG'].release_GPU(app.config[f'gpu:{n}'])
                # Charge le nouveau modèle en mémoire
                app.config[f'gpu:{n}'] = setup_model(app.config['RAG'], n)
                
        except :
            # Sécurité si le nouveau chemin du modèle n'est pas bon, renvoie l'ancien
            app.config['RAG'].model_path = last_name
            for n in app.config['gpu'].get_gpu() :
                # Libère la mémoire GPU des anciens modèles 
                app.config['RAG'].release_GPU(app.config[f'gpu:{n}'])
                # Recharge l'ancien modèle en mémoire
                app.config[f'gpu:{n}'] = setup_model(app.config['RAG'], n)
            
            return jsonify({"error": "The model path doesn't exist or is too big"}), 400


    if json_file.get("server"):
        print(app.config['gpu'].get_gpu(),json_file["server"]["gpu"])
        
        for n in set(app.config['gpu'].get_gpu()+json_file["server"]["gpu"]) :
            # Si d'anciens GPUs n'apparaissent plus dans le json, les libère
            if n in app.config['gpu'].get_gpu() and n not in json_file["server"]["gpu"] : 
                print('1')
                app.config['RAG'].release_GPU(app.config[f'gpu:{n}'])

            # Si de nouveaux GPUs apparaissent dans le json, charge le modèle en mémoire
            if n in json_file["server"]["gpu"] and n not in app.config['gpu'].get_gpu() :
                print('2')
                app.config[f'gpu:{n}'] = setup_model(app.config['RAG'], n) 
        app.config['gpu'].change_gpu(json_file["server"]["gpu"])


    if json_file.get("RAG"):
        app.config['RAG'].change_parameters(json_file["RAG"])

    if json_file.get("scheduler"):

        update_scheduler(app,
                        app.config['Scheduler'],
                        json_file['scheduler']['day_period'],
                        json_file['scheduler']['hour'],
                        json_file['scheduler']['minute']) 
        
    modify_json(config_path,json_file)
    
    return jsonify({}), 204

