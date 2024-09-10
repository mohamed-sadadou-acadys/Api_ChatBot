'''Ce fichier contient les fonctions principales pour la gestion de la base de données du Campus Manager côté gestion des conversations. Elle contient entre autres des fonctions pour l'extraction et le stockage de données'''

from pymongo import MongoClient
from bson.objectid import ObjectId
import json

class Database():
    def __init__(self):
        
        with open('API/authentification.json') as f:
            authFile = json.load(f)

        # Authentification au client MongoDB
        myClient = MongoClient(f"mongodb+srv://{authFile['username']}:{authFile['password']}@v0.84y7vus.mongodb.net/?retryWrites=true&w=majority"
                                    )
        
        print(myClient.test_database)

        self.database = myClient["Campus_Manager"] # Base de données
        self.userCollection = self.database["users"] # Collection des utilisateurs
        self.formationCollection = self.database["formations"] # Collection des formations



    def createConversation(self,id_user,question,date,title,bot_response,context_RAG):  
        '''
        Créer une nouvelle conversation à partir d'une réponse générée. 
        '''

        conversationToCreate =  {
                                "_id":ObjectId(),
                                "title": title,
                                "last_update": date,
                                "context_RAG": context_RAG,
                                "notation": 0,
                                "comment": "",
                                "remark": "", 
                                "n_token": {
                                    "input": 0,
                                    "output": 0
                                },
                                "Exchanges": [
                                    {
                                        "question": question,
                                        "answer": bot_response
                                    }
                                ],
                                "Expert":{}
                                }
        
        # Ajout de l'objet conversation créé dans la liste à la première position (derniers messages toujours en premier)
        result = self.userCollection.update_one({'_id': ObjectId(id_user)}, 
                                                {"$inc": {"nb_conv":1},
                                                "$push": {"Conversations": {"$each": [conversationToCreate], 
                                                                               "$position": 0}}})
        
        return result.matched_count, str(conversationToCreate["_id"])
    


    def addExchanges(self,id_user,i_conv,question,bot_response,date,context_RAG,conversations):
        '''
        Ajoute un nouvel échange à une conversation existante à partir d'une réponse générée. 
        '''

        exchangeToAdd = {
                        "question": question,
                        "answer": bot_response
                        }

        # Récupération et suppression de la conversation existante
        conversation = conversations.pop(i_conv)

        # Mise à jour de la conversation
        conversation['context_RAG'] = context_RAG
        conversation['last_update'] = date
        conversation['Exchanges'].insert(0,exchangeToAdd)

        # Ajout de la conversation dans la liste à la première position (derniers messages toujours en premier)
        conversations.insert(0,conversation)

        # Mise à jour de l'objet conversations dans la base de données                       
        result = self.userCollection.update_one(
                                        {"_id": ObjectId(id_user)},
                                        {"$set": {"Conversations": conversations}}
                                    )

        return result.matched_count
    

    def getObjectsInConv(self,id_user,id_conv,object_keys):
        '''
        Recherche des objets dans une conversation. 
        '''
        
        # L'opérateur $ permet de ressortir un seul objet à l'intérieur de Conversations
        objects_to_find = { f"Conversations.{object_keys[0]}.$": 1 }

        # Ajout de l'ensemble des objets à rechercher dans Conversations
        for object_key in object_keys[1:] : 
            objects_to_find[f"Conversations.{object_key}"] = 1
                
        objects_found = self.userCollection.find_one({'_id': ObjectId(id_user),
                                            'Conversations._id': ObjectId(id_conv)}, 
                                                   objects_to_find)
        
        return objects_found
        

    def getObjectsForUser(self,id_user,object_keys):
        '''
        Recherche des objets pour un utilisateur.
        '''

        # Ajout de l'ensemble des objets à rechercher pour un utilisateur
        objects_to_find = { f"Conversations.{object_key}": 1 for object_key in object_keys}

        objects_found = self.userCollection.find_one({'_id': ObjectId(id_user)},
                                                    objects_to_find)
        
        return objects_found
    
    
    def getObjects(self,object_keys):
        '''
        Recherche des objets pour tous les utilisateurs.
        '''

        # Ajout de l'ensemble des objets à rechercher dans la base de données
        objects_to_find = { f"Conversations.{object_key}": 1 for object_key in object_keys}

        objects_found = self.userCollection.find({}, objects_to_find)
        
        return objects_found
    

    def getFormation(self,id_formation):
        '''
        Recherche une formation dans la base de données de formation. 
        '''

        formation = self.formationCollection.find_one({"_id" : ObjectId(id_formation)})
        return formation


    def getAllUser(self):
        '''
        Récupère tous les utilisateurs. 
        '''

        users = self.userCollection.find({}, {"_id": 1})
        return  [str(user['_id']) for user in users]


    def getIdsFormation(self,id_user):
        '''
        Recherche les IDs de formation permis à un utilisateur. 
        '''

        ids_formation = self.userCollection.find_one({"_id" : ObjectId(id_user)},
                                                    {"formations": 1})
        
        return ids_formation


    def getAllConversations(self,id_user):
        '''
        Récupère l'ensemble des conversations d'un utilisateur.
        '''
        
        conversations = self.userCollection.find_one({"_id" : ObjectId(id_user)},
                                                    {"Conversations": 1})


        return conversations
    

    def getManyConversations(self,id_user,i_start,i_end):
        '''
        Récupère les conversations d'un utilisateur entre les indices i_start et i_end. 
        '''

        conversations = self.userCollection.find_one(
                                    {"_id" : ObjectId(id_user)},
                                    {"Conversations":{"$slice": [i_start,i_end]},
                                    "nb_conv":1}
                                    )
        return conversations


    def getOneConversation(self,id_user,id_conv):
        '''
        Récupère une conversation d'un utilisateur. 
        '''
        
        conversations = self.userCollection.find_one(
                                    {"_id" : ObjectId(id_user),
                                    "Conversations._id":ObjectId(id_conv)},
                                    {"Conversations.$":1})
        return conversations


    def changeIDconvToI(self,conversations,ids):
        '''
        Echange l'ID de la conversation (id_conv) en indice de l'objet conversations
        '''

        # Trouver le premier objet avec l'ID recherché et relève son indice
        is_to_add = []
        for i, conv in enumerate(conversations):
            if str(conv["_id"]) in ids:
                is_to_add.append(i)

        
        return is_to_add



    def updateObjectsInConv(self,object_keys,id_user,id_conv,object_values):
        '''
        Mets à jour les objets d'une conversation. 
        '''

        # Objets à mettre à jour et nouvelle valeurs 
        objects_to_set = {
            f"Conversations.$.{object_key}" : object_value for object_key,object_value in zip(object_keys,object_values)
         }
        
        result = self.userCollection.update_one({'_id': ObjectId(id_user),
                                            'Conversations._id': ObjectId(id_conv)}, 
                                                    {'$set': objects_to_set})
        
        return result.matched_count



    def deleteConversation(self,id_user,ids_conv,conversations):
        '''
        Supprime une conversation.
        '''

        # Triage de la liste d'indice à supprimer dans le sens inverse
        ids_conv.sort(reverse=True)

        # Suppression des éléments sans problème de mise à jour
        for id_conv in ids_conv:
            del conversations[id_conv]
        
        result = self.userCollection.update_one({'_id': ObjectId(id_user)}, 
                                                 {"$inc": {"nb_conv":-len(ids_conv)},
                                                '$set':{"Conversations":conversations}
                                                 })
        
        return result.matched_count*len(ids_conv)



if __name__=='__main__':

    db = Database()
    db.getAllUser()

    
