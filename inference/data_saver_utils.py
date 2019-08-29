import pandas as pd
import numpy as np
def store_gathered_data(collected_data, output_path="collected_data_raw.csv"):
    "this function saves the gathered data to a data frame, so we can do stuff with if it"
    as_df = pd.DataFrame(collected_data)
    as_df.columns = ["frame", "age", "gender", "emotion", "yaw", "pitch", "roll"]
    print("[INFO] saving raw collected data to csv in in path:", output_path)
    as_df.to_csv(output_path)
    return as_df 

def one_hot_encode_gathered_data(df, output_path="collected_preprocessed_data.csv"):    
    "this function scales and onehot encodes the gathered data so we can do machine learning with it - also saves it in a csv)"
    male = df["gender"] == "male"
    female = df["gender"] != "male"

    df = pd.concat([df, male], axis=1)
    df = pd.concat([df, female], axis=1)

    happy = df["emotion"] == "happy"
    sad = df["emotion"] == "sad"
    anger = df["emotion"] == "anger"
    neutral = df["emotion"] == "neutral"
    surprise = df["emotion"] == "surprise"

    df = pd.concat([df, happy], axis=1)
    df = pd.concat([df, sad], axis=1)
    df = pd.concat([df, anger], axis=1)
    df = pd.concat([df, neutral], axis=1)
    df = pd.concat([df, surprise], axis=1)


    df.columns = ["frame", "age", "gender", "emotion", "yaw", "pitch", "roll", "male", "female", "happy", "sad", "anger", "neutral", "surprise"]


    df = df.drop("gender", axis=1)
    df = df.drop("emotion", axis=1)
    
    #scaling, because the maximum age is 100 for the neuralnet
    #df["age"] = df["age"]/100

    
    #scaling and normalizing (X-Mean)/std
    df["age"] = (df["age"] - np.mean(df["age"]))/ np.std(df["age"])
    df["yaw"] = (df["yaw"] - np.mean(df["yaw"]))/ np.std(df["yaw"])
    df["roll"] = (df["roll"] - np.mean(df["roll"]))/ np.std(df["roll"])
    df["pitch"] = (df["pitch"] - np.mean(df["pitch"]))/ np.std(df["pitch"])
    print("[INFO] saving preprocessed data to csv in in path:", output_path)
    df.to_csv(output_path)
    return df