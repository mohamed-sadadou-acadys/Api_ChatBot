'''Ce fichier contient les fonctions principales pour la gestion de l'application FLASK du Campus Manager, dont le générateur et le transmetteur des tokens générés.'''

import queue 
import atexit
import threading
from RAG.scripts.Generation import RAG, StoppingCriteriaList,StoppingCriteriaSub
from API.utils import actual_datetime, change_to_datetime, timedelta, shorten_datetime
from RAG.scripts.Preprocessing import Preprocessing
from RAG.scripts.ExternalPreprocessing import ExternalProcessing
import time


class server_gpu():
    '''
    La classe qui modèlise la liste des serveurs gpus utilisés pour load les modèles. 
    '''

    def __init__(self,gpu):
        self.gpu = gpu # Liste des gpus à load

    def get_gpu(self):
        return self.gpu
    
    def change_gpu(self,gpu_to_use):
        self.gpu=gpu_to_use



class ThreadedGenerator:
    ''' 
    Le générateur de token récupère les tokens générés par le LLM dans une queue pour pouvoir les transmettre ensuite à l'utilisateur de façon asynchrone
    '''

    def __init__(self):
        self.queue = queue.Queue()
        self.is_running = False

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()

        # L'item d'arrêt indique une fin de génération
        if item is StopIteration: 
            raise item

        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        print("CLOSE")
        self.is_running = False
        self.queue.put(StopIteration)
    


def setup_model(myRAG,device):
    '''
    Charge le modèle sélectionné sur le device correspondant
    '''

    print("device : ",device)

    g = ThreadedGenerator()

    # Chargement du modèle en GPU
    model, tokenizer = myRAG.get_model_for_GPU(device=device)

    # Liste des mots en token qui stoppent la génération
    stop_words = myRAG.stop_sequence_list
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_ids)])


    return {'generator' : g,
            'model' : model, 
            'tokenizer' : tokenizer, 
            'stopping_criteria':stopping_criteria,
            'currentSid' : ''}


def send_wait(app,socketio):
    '''
    Envoie un message wait toutes les 25 secondes aux utilisateurs dans la queue
    '''

    while True:
        # Attends 25 secondes de manière asynchrone
        socketio.sleep(25) 
        for requestInQueue in app.config['queueRequest']:
            socketio.emit('event', {'message': 'wait'}, room=requestInQueue["sid"])


def thread_chatbot(app,socketio):   
    '''
    Lance le chatbot en arrière plan. 
    Lorsqu'une nouvelle requête arrive dans la file d'attente, la fonction sélectionne la première GPU disponible et lance la génération dessus.
    '''

    while True : 
        # Check de la file d'attente toutes les 3 secondes 
        time.sleep(3)

        # La file d'attente n'est pas vide 
        if len(app.config['queueRequest']) > 0:
            print('ok_chat',len(app.config['queueRequest']), app.config['queueRequest'])

            # Check de la disponibilité des GPUs
            for n in app.config["gpu"].get_gpu() : 
                print("gpu is running : ",n,app.config[f'gpu:{n}']['generator'].is_running)
                
                # Une GPU est disponible (elle n'est pas en cours d'utilisation)
                if app.config[f'gpu:{n}']['generator'].is_running==False :

                    # L'état de la GPU est changé en cours d'utilisation 
                    app.config[f'gpu:{n}']['generator'].is_running=True

                    # La requête est extraite de la file d'attente
                    currentRequest = app.config['queueRequest'].pop(0)

                    # Le SID de l'utilisateur est conservé 
                    app.config[f'gpu:{n}']['currentSid'] = currentRequest['sid']
                    print(n,currentRequest["question"])

                    # Le chat est nouveau, il n'y a pas d'historique
                    if currentRequest["event"] == "create_chat":
                        history,conversations,i_to_add = [], None, None

                    # Le chat est ancien, il y a un historique à récupérer 
                    elif currentRequest["event"] == "add_chat":
                        history,conversations,i_to_add = search_conversation(app,
                                                                            socketio,
                                                                            app.config[f'gpu:{n}']['currentSid'],
                                                                            currentRequest["id_user"],
                                                                            currentRequest["id_conv"])
                        
                    # Récupération des ids de formation pour le RAG permis à l'utilisateur
                    ids_formations = search_ids_formation(app,
                                                        socketio,
                                                        app.config[f'gpu:{n}']['currentSid'],
                                                        currentRequest["id_user"])       
                    print('OK GO SEND')

                    # Lancement de la génération et du stockage de la conversation en arrière plan
                    threading.Thread(target=thread_stream_and_stock, 
                                    args=(app,
                                            socketio,
                                            app.config[f'gpu:{n}']['currentSid'],
                                            currentRequest["event"],
                                            currentRequest["id_conv"],
                                            currentRequest["id_user"],
                                            currentRequest["question"],
                                            conversations,
                                            ids_formations,
                                            history,
                                            i_to_add,
                                            n,)).start()

                    print('break')
                    # La requête a été transmise à une GPU, on sort de la boucle 
                    break



def search_conversation(app,socketio,sid,id_user,id_conv):
    '''
    Cherche une conversation existante d'un utilisateur et renvoie l'historique. 
    '''

    # Récupère l'ensemble des conversations
    conversations_dict = app.config['Database'].getAllConversations(id_user)

    if conversations_dict is None :
        with app.app_context():  
            socketio.emit('event', {'message': 'id_user not in database'}, room= sid)
        return None, None, None


    if conversations_dict["Conversations"] == [] :
        with app.app_context():  
            socketio.emit('event', {'message': 'id_conv not in database'}, room= sid)
        return None, None, None
    
    # Transforme l'ID de la conversation en indice de la variable conversations
    is_to_add = app.config['Database'].changeIDconvToI(conversations_dict["Conversations"],
                                                                    id_conv)
    
    # Récupère les messages de la conversation recherchée pour créer l'historique
    conv_exchanges = conversations_dict["Conversations"][is_to_add[0]]['Exchanges']
    history = [(conv_exchanges[i]['question'].replace("\\n", "\n"),
             conv_exchanges[i]['answer'].replace("\\n", "\n")) for i in range(len(conv_exchanges))]

    return history, conversations_dict["Conversations"], is_to_add[0]


def search_ids_formation(app,socketio,sid,id_user):
    '''
    Recherche les ids de formations permis à l'utilisateur pour le RAG
    '''

    ids_formations_dict = app.config['Database'].getIdsFormation(id_user)

    if ids_formations_dict is None :
        with app.app_context():  
            socketio.emit('event', {'message': 'id_user not in database'}, room= sid)
        return None
    
    return list(map(str,ids_formations_dict["formations"]))
    


def thread_stream_and_stock(app,socketio,sid,event,id_conv,id_user,question,conversations,ids_formations,history,i_to_add,n):
    '''
    Lance la génération, l'envoie et le stockage des messages générées dans la base de données en arrière plan.
    '''

    # Critère d'arrêt mis en OFF pour commencer la génération
    app.config[f'gpu:{n}']['stopping_criteria'].change_to_False()   

    # Génération du RAG sur la GPU correspondante
    list_ids_RAG,n_token = app.config['RAG'].run_for_gpu(app.config[f'gpu:{n}']['model'],
                                    app.config[f'gpu:{n}']['tokenizer'],
                                    question,
                                    app.config[f'gpu:{n}']['generator'],
                                    history=history,
                                    stopping_criteria = app.config[f'gpu:{n}']['stopping_criteria'],
                                    WordEmbedding=app.config['WordEmbedding'],
                                    ExternalWordEmbedding=app.config['ExternalResourcesEmbedding'],
                                    ids_formation=ids_formations)
    

    bot_response = ""


    # Tant qu'il y a des tokens générés dans la queue, transmets le token à l'utilisateur correspondant
    for item in app.config[f'gpu:{n}']['generator'] :

        n_token['output'] +=1

        print("item : ",item,", device : ",n, "sid : ",sid)

        bot_response += item.replace("\n", "\\n")

        with app.app_context():  

            # Transmission du token à l'utilisateur
            socketio.emit('chat_token', 
                            {'event': event, 'token': item.replace("\n", "\\n"),'id_conv':id_conv}, 
                            room=sid)
        print("item sent")

    print(bot_response)


    title = "Nouveau chat"
    date = actual_datetime().isoformat() # sous format string


    if event == "create_chat":
        # Stocke le message de l'utilisateur dans un nouveau chat 
        updated_count,id_conv = app.config['Database'].createConversation(id_user,
                                                                        question.replace("\n", "\\n"),
                                                                        date,
                                                                        title,
                                                                        bot_response,
                                                                        list_ids_RAG)


        if updated_count < 1 :
            with app.app_context():  
                socketio.emit('event', {'message': 'id_user not in database'}, 
                                        room=sid)
            return 
        
        with app.app_context(): 
            # Transmission du titre de la conversation et de la date à l'utilisateur 
            socketio.emit('chat_data', {'event': event,'date':date,'title':title,'id_conv':id_conv}, room= app.config[f'gpu:{n}']['currentSid'])

    elif event == "add_chat":
        # Stocke le message de l'utilisateur dans un chat existant   
        _ = app.config['Database'].addExchanges( id_user,
                                                    i_to_add,
                                                    question.replace("\n", "\\n"),
                                                    bot_response,
                                                    date,
                                                    list_ids_RAG,
                                                    conversations)
        with app.app_context(): 
            # Transmission du titre de la conversation et de la date à l'utilisateur 
            socketio.emit('chat_data', {'event': event,'date':date}, 
                          room=sid)

    # Mise à jour du nombre de token de la conversation
    store_n_token(app,n_token,id_user,id_conv)




def store_n_token(app,n_token,id_user,id_conv):
    '''
    Mets à jour le nombre de token de la conversation dans la base de données. 
    '''

    # Nombre de token stocké initialement
    previous_n_token = app.config['Database'].getObjectsInConv(id_user,
                                                    id_conv,
                                                    ["n_token"])["Conversations"][0]["n_token"]

    
    print(previous_n_token)
    print(n_token)

    # Ajout du nombre de token du nouveau message
    new_n_token = {
        cle: previous_n_token.get(cle, 0) + n_token.get(cle, 0)
        for cle in set(previous_n_token) | set(n_token)
    }
    
    print(new_n_token)

    # Mise à jour du nombre de token dans la base de données 
    _ = app.config['Database'].updateObjectsInConv(["n_token"],
                                                           id_user,
                                                           id_conv,
                                                           [new_n_token])



def preprocess_new_data(path):
    '''
    Prétraite les documents de formation pour les transformer en DataFrame. 
    Le texte est découpé en fonction du type de document. 
    '''

    preprocessing = Preprocessing()
    print("le path a traité dans preprocess_new_data")
    print(path[path.rfind('.')+1:])
    # Extraction et découpage pour des PowerPoints
    if path[path.rfind('.')+1:] in ['ppt','pptx']:
        df_formation = preprocessing.extract_text_pptx(ppt_url=path)

    # Extraction et découpage pour des Docs
    elif path[path.rfind('.')+1:] in ['doc','docx']: 
        df_formation = preprocessing.extract_text_docx(doc_url=path)

    # Extraction et découpage pour des PDFs
    elif path[path.rfind('.')+1:] in ['pdf']: 
        df_formation = preprocessing.extract_text_pdf(pdf_url=path)
    
    # Suppression des lignes avec des cases nulles
    df_formation = preprocessing.remove_nan_values(df_formation)

    # Suppression des lignes contenant moins d'un certain nombre de mots (trop peu de texte important)
    df_formation = preprocessing.remove_small_texts(df_formation)

    return df_formation

def preprocess_external_data(path):
    '''
    Preprocesses a document (Word or PDF) to transform it into a DataFrame.
    The text is extracted and structured based on the document's layout.
    '''
    # Initialize the external processing class
    external_processing = ExternalProcessing()

    # Check the file extension and ensure it's either a Word document or PDF
    file_extension = path[path.rfind('.')+1:].lower()
    
    if file_extension not in ['docx', 'pdf']:
        raise ValueError("This function only supports documents with .docx or .pdf extensions.")

    # Extraction and structuring based on the file type
    if file_extension == 'docx':
        df_formation = external_processing.extract_text_from_docx(docx_url=path)
    elif file_extension == 'pdf':
        print("on a un pdf")
        df_formation = external_processing.extract_text_from_pdf(pdf_url=path)

    # If needed, remove rows with NaN or small texts here
    df_formation = df_formation.dropna().reset_index(drop=True)

    return df_formation




def delete_conversations_by_date(app,day_period,hour,minute):
    '''
    Supprime les conversations de la base de données qui dépasse la période de temps. 
    '''
    
    # La plus ancienne date après laquelle les messages sont supprimées 
    oldest_day = shorten_datetime(actual_datetime()) - timedelta(days = day_period)

    ids_user = app.config['Database'].getAllUser()
    deleted_count = 0

    # Inspection de l'ensemble des conversations de tous les utilisateurs 
    for id_user in ids_user :
        conversations = app.config['Database'].getAllConversations(id_user)['Conversations']
        id_conv_to_delete = []

        for id_conv,conversation in enumerate(conversations) :
            
            # Si la plus ancienne date est supérieur à la conversation, elle sera supprimée
            if change_to_datetime(conversation['last_update']) < oldest_day:
                id_conv_to_delete.append(id_conv)

        # Supprime les conversations les plus anciennes de l'utilisateur
        _ = app.config['Database'].deleteConversation(id_user,
                                                    id_conv_to_delete,
                                                    conversations)
        
        deleted_count += len(id_conv_to_delete)         

    print(f'n_deleted : {deleted_count}')


def initialize_scheduler(app,scheduler,day_period,hour,minute):
    '''
    Initialise le scheduler pour supprimer les conversations de plus d'un an tous les jours à midi. 
    '''

    scheduler.add_job(delete_conversations_by_date, 
                            'cron', 
                            args=(app, day_period, hour, minute), 
                            hour=hour, 
                            minute=minute,
                            id = 'delete_conversation') 
    
    scheduler.start()

    # Arrête le scheduler lorsque l'application se termine
    atexit.register(lambda: scheduler.shutdown())


def update_scheduler(app,scheduler,day_period,hour,minute):
    '''
    Mets à jour le scheduler pour supprimer les conversations dépassant la période de temps tous les jours à midi. 
    '''

    # Supprime le scheduler actuel avant d'en relancer un nouveau
    scheduler.remove_job('delete_conversation')

    scheduler.add_job(delete_conversations_by_date, 
                                    'cron', 
                                    args=(app, 
                                        day_period, 
                                        hour, 
                                        minute), 
                                    hour=hour, 
                                    minute=minute,
                                    id = 'delete_conversation') 



def merge_n_token(all_n_token_object,date1,date2):
    '''
    Somme les nombres de token du dictionnaire en fonction de leurs clés (input, output, id_formation). 
    '''

    token_sum = {}

    # Pour chaque utilisateur
    for n_token_object in all_n_token_object : 

        # Pour chaque conversation 
        for n_token_set in n_token_object['Conversations']:
            
            for key, value in n_token_set['n_token'].items():

                if date1 and date2: 
                    # Si la date de la conversation est comprise dans la période recherchée
                    if change_to_datetime(date1) < change_to_datetime(n_token_set['last_update']) and change_to_datetime(n_token_set['last_update']) < change_to_datetime(date2) :
                        # Ajouter ou mettre à jour le nombre de token dans le dictionnaire fusionné
                        token_sum[key] = token_sum.get(key, 0) + value

                else :
                    token_sum[key] = token_sum.get(key, 0) + value

    return token_sum

