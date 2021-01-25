import logging
import ast
from azure.storage.blob import ContainerClient
from azure.storage.blob import BlobClient
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

import azure.functions as func
import os
from io import BytesIO
import pandas as pd
import json
import base64
import requests
def main(msg: func.QueueMessage) -> None:
    logging.info('Python queue trigger function processed a queue item: %s',
                 msg.get_body().decode('utf-8'))
    body= msg.get_body().decode('utf-8')
    json_content = ast.literal_eval(body)  
    data_content = json_content['data']
    url = data_content['url']
    print(url)
    feature_file_name='sample_feature.json'
    blob_name= "/".join(url.split("/")[-3:])
    feature_file_blob_name =  "/".join(url.split("/")[-3:-1])+"/" +feature_file_name
    print("blob name is ", blob_name)
    print("feature_file_blob_name is ", feature_file_blob_name)

    credentials = ManagedIdentityCredential()
    secret_client = SecretClient(vault_url="https://databricksencryption.vault.azure.net", credential=credentials)
    conn_str = secret_client.get_secret("connectionstringazurefunction")
    conn_str=conn_str.value
    print("conection string ", conn_str)
    container_name="aml-mlflow-object-detection"
    container = ContainerClient.from_connection_string(conn_str=conn_str, container_name=container_name)
    blob = BlobClient.from_connection_string(conn_str=conn_str, container_name=container_name, blob_name=blob_name)
    
    folder= os.path.join("/tmp", "/".join(blob_name.split("/")[:-1]))
    feature_file_blob =BlobClient.from_connection_string(conn_str=conn_str, container_name=container_name, blob_name=feature_file_blob_name)

    os.makedirs(folder, exist_ok=True)
    with open(os.path.join("/tmp",blob_name), "wb") as my_blob:
        download_stream= blob.download_blob()
        download_stream.readinto(my_blob)

    with open(os.path.join("/tmp",feature_file_blob_name), "wb") as my_feature:
        download_stream= feature_file_blob.download_blob()
        download_stream.readinto(my_feature)

    print("content of download:", os.listdir(folder))



    ENCODING = 'utf-8'
    test_image_paths =[os.path.join("/tmp",blob_name)]
    additional_feature_file =os.path.join(folder,feature_file_name)
    print("additional image file final path ",additional_feature_file)
    feature_file_df = pd.read_json(additional_feature_file)
    
    base64_string_list=[]
    for test_image_path in test_image_paths:
        with open(test_image_path,"rb") as img:
            image_bytes = BytesIO(img.read())
        encoded_image =base64.b64encode(image_bytes.getvalue())
        base64_string = encoded_image.decode(ENCODING)
        base64_string= "b'{0}'".format(base64_string)
        base64_string_list.append(base64_string)
    image_request = {"image": base64_string_list, "additional_feature":[feature_file_df.iloc[0].to_json()]}
    image_request= {"data": image_request}
    # payload.append(image_request)

    body = json.dumps(image_request)


    scoring_uri ='http://486cc62b-8960-45f2-aa93-76cd3d844d56.westus2.azurecontainer.io/score'

    # headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
    headers = {'Content-Type':'application/json'}

    # uri ='http://9c729eb8-c2d1-4f24-9114-fbd35d040d7e.westus2.azurecontainer.io/score'
    response = requests.post(url=scoring_uri, data=body,headers=headers)
    print(response.text)

    

    #Trigger Model scoring which is an API hosted in AKS
    #here is the URI to the folder, process it.
    #API returns the result which is the json
    #get json and store the json in Cosmos
    #Call another Azure Function (set a flag in a queue( i am done))
    #Indepedently, another queue triggered /Cosmos triggered azure function, Post-Processing Azure Function (reading result from Cosmos), store files in ADLS 

