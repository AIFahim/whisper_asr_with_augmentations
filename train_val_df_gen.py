import glob
import pandas as pd
import codecs
import json


class Train_Val_df:
    # def __init__(self):
    #     self.generate_df_from_json()

    def processing_dataset(self , text):
            """ Use a phone map and convert phoneme sequence to an integer sequence """
            phone_list = text.split('_2')
            phone_list.pop()
            int_sequence = []
            # print(phone_list)

            sentence = ""
            for phone_per_word in phone_list:
                phone_per_word = phone_per_word.lstrip()
                phone_per_word = phone_per_word.replace("_1", "")
                phone_per_word = phone_per_word.replace(" ", "")
                sentence += phone_per_word
                sentence += "@"
            return sentence
    
    @staticmethod
    def generate_df_from_dir(self):
        flac_path_train = glob.glob('/home/asif/Datasets_n_models_checkpoints/dataset_grapheme_cleaned/train/*.flac')
        flac_path_val = glob.glob("/home/asif/Datasets_n_models_checkpoints/dataset_grapheme_cleaned/valid/*.flac")
        # txt_path_train = glob.glob("/content/dataset/train/*.txt")
        # txt_path_val = glob.glob("/content/dataset/valid/*.txt")
        
        txt_list_train = []
        txt_list_val = []
        train_df = pd.DataFrame()
        for i in flac_path_train:
            # print(i)
            # print("///")
            i = i.replace("flac","txt")
            # print(i)
            with codecs.open(i, 'r', 'utf-8') as fid:
                for line in fid.readlines():
                    # processed_line = self.processing_dataset(line)
                    # txt_list_train.append(processed_line)
                    txt_list_train.append(line)
        

        for i in flac_path_val:
            i = i.replace("flac","txt")
            with codecs.open(i, 'r', 'utf-8') as fid:
                for line in fid.readlines():
                    # processed_line = self.processing_dataset(line)
                    # txt_list_val.append(processed_line)
                    txt_list_val.append(line)


        # flac_path_train.sort()

        data_train = {
            'audio': flac_path_train,
            'sentence':txt_list_train,
        }

        data_val = {
            'audio': flac_path_val,
            'sentence':txt_list_val,
        }
        train_df = pd.DataFrame(data_train)
        val_df = pd.DataFrame(data_val)
        return train_df , val_df


    @staticmethod
    def generate_df_from_json(json_data, train_ratio=0.9):
        flac_path = []
        txt_list = []
        cnt = 1
        
        # print("--------------len(json_data)-------------",len(json_data))
        for key, value in json_data.items():


            audio_path = value['audio']

            # print("--------audio_path----------",audio_path)

            audio_path = f"/UPDS/STT/large_dataset_reve_cv_oslr/"+audio_path

            # print(audio_path)
            text = value['text']

            flac_path.append(audio_path)
            txt_list.append(text)

            cnt+=1
        # print(flac_path[:10])
        flac_path = sorted(flac_path)
        txt_list = sorted(txt_list)

        # print(" len(flac_path) ",len(flac_path))
        # print(" len(txt_list) ",len(txt_list))
        # print(flac_path[:10])


        data = {
            'audio': flac_path,
            'sentence': txt_list,
        }

        df = pd.DataFrame(data)

        # Shuffle the DataFrame
        # df = df.sample(frac=1).reset_index(drop=True)

        # Calculate the split index
        split_index = len(df) - 2 # int(train_ratio * len(df))

        # Split the DataFrame into train and validation DataFrames
        train_df = df[:split_index]
        val_df = df[split_index:]

        return train_df, val_df #, split_index


if __name__ == "__main__":
    # Read the JSON file
    with open('/home/asif/merged_dict.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Call the generate_df_from_json() method on the Train_Val_df class directly
    train_df, val_df, si = Train_Val_df.generate_df_from_json(json_data)

    print(len(train_df))
    # print(val_df)

