import pymongo
import requests,json
import pandas as pd

client = pymongo.MongoClient("INSERT MONGODB CONNECTION LINK")

if __name__ == "__main__":
    # auth
    token_url = "https://www.promet.si/uc/user/token"
    data = {'grant_type': 'password',
            'username': 'EMAIL',
            'password': 'PASSWORD'
            }

    access_token_response = requests.post(token_url, data=data, headers={"Content-Type":"application/x-www-form-urlencoded"})
    tokens = json.loads(access_token_response.text)

    # query
    test_api_url = "https://www.promet.si/dc/b2b.stevci.geojson.sl_SI"
    api_call_headers = {'Authorization': 'bearer ' + tokens['access_token']}
    api_call_response = requests.get(test_api_url, headers=api_call_headers)

    data = json.loads(api_call_response.text)

    df=pd.json_normalize(data['features'])
    df=df[(
            df['properties.title'].str.contains('južna') |
            df['properties.title'].str.contains('S obvoznica') |
            df['properties.title'].str.contains('vzh. obvoznica') |
            df['properties.title'].str.contains('LJjug-LJBrdo') |
            df['properties.title'].str.contains('LJBrdo-LJjug') |
            df['properties.title'].str.contains('Brdo - Kozarje') |
            df['properties.title'].str.contains('Kozarje - Brdo') |
            df['properties.title'].str.contains('Brdo-LJsev') |
            df['properties.title'].str.contains('LJsev-Brdo') |
            df['properties.title'].str.contains('Celovška - Koseze') |
            df['properties.title'].str.contains('Celovška - Dunajska') |
            df['properties.title'].str.contains('Priključek Nove Jarše') |
            df['properties.title'].str.contains('LJsev-LJvzh') |
            df['properties.title'].str.contains('LJvzh-LJsev') |
            df['properties.title'].str.contains('LJvzh-LJjug') |
            df['properties.title'].str.contains('LJjug-LJvzh')
           )]

    # rename cols
    df.columns=['type','geometry','coordinates','title','id','date','summary','stevci_lokacija','stevci_lokacijaOpis','stevci_cestaOpis','stevci_odsek','stevci_stacionaza','stevci_smer','stevci_smerOpis','stevci_pasOpis','stevci_regija','stevci_geoX','stevci_geoY','stevci_vmax','stevci_datum','stevci_ura','stevci_stev','stevci_hit','stevci_gap','stevci_occ','stevci_stat','stevci_statOpis']
    df = df[['coordinates', 'title', 'id', 'summary', 'stevci_lokacija', 'stevci_lokacijaOpis', 'stevci_cestaOpis',
             'stevci_odsek', 'stevci_smer', 'stevci_smerOpis', 'stevci_pasOpis', 'stevci_vmax', 'stevci_datum',
             'stevci_ura', 'stevci_stev', 'stevci_hit', 'stevci_gap', 'stevci_occ', 'stevci_statOpis']]

    # save to database
    db = client["DB"]
    collection = db["COLLECTION"]

    data_dict = df.to_dict("records")
    response = collection.insert_many(data_dict)
