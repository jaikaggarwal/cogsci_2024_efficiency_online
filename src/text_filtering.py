if __name__ == '__main__':

    for year in range(2019, 2020):
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            # Load selected raw data
            raw = pd.read_json(f"{RAW_DATA}RC_{year}-{month}.json", lines=True)
            
            # Find and replace amp with blank
            for index, sentence in enumerate(raw['body']):
                fixed = sentence.replace("&amp;", "")
                raw['body'][index] = fixed


"""
            # Template
            for index, sentence in enumerate(raw['body']):
                sentence = sentence.replace("FILL IN", "")
                raw['body'][index] = sentence
"""

