import pickle
import pandas as pd
import io
import requests

def test_prediction(url, model):

    return 'prediction'

if __name__ == "__main__":
    # df = pd.read_pickle('https://github.com/user/mydirectoryname/raw/main/Results/mypicklefile')

    # Load Pipeline from pickle file
    with open("wg_price_predictor/models/Pred_pipeline_WG_allcities.pkl", 'rb') as pfile:
        model=pickle.load(pfile)

    download = requests.get('https://raw.githubusercontent.com/chvieira2/housing_crawler/master/housing_crawler/data/ads_OSM.csv').content

    df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    print(model.predict([df.head(1)]))
    print(test_prediction('url', model))
