lsimport pandas as pd
import time
import glob
from nltk.corpus import stopwords
import string
from urllib.request import urlopen

'''
script to download images from KB
'''

stoplist = stopwords.words('dutch')


def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s


def load_file():
    path = '../data/'
    allFiles = glob.glob(path + "*.tsv")
    print(allFiles)

    for f in allFiles:
        print(f)
        df = pd.read_csv(f, delimiter='\t', parse_dates=True)
        df = df.dropna(subset=['ocr'])
        # remove files that contain error msg
        excludes = ['objecttype', 'file directory not found']
        df = df[~df['ocr'].astype(str).str.contains('|'.join(excludes))]
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['id'] = df['ocr_url'].apply(lambda x: x[38:-4])
        df['ocr'] = df['ocr'].str.findall('\w{4,}').str.join(' ')  # remove words shorter than 4 characters
        df['ocr'] = df['ocr'].apply(lambda S:S.strip('Advertentie '))  # remove initial Advertentie
        df['ocr'] = df['ocr'].str.replace('\d+', '')  # remove digits
        df['ocr'] = df['ocr'].str.strip()
        df['ocr'] = [' '.join(filter(None,filter(lambda word: word not in stoplist, line))) for line in df['ocr'].str.lower().str.split(' ')]
        df['ocr'] = df['ocr'].apply(remove_punctuation)  # remove punctuation
        df['area'] = df['w'] * df['h']

        # remove error
        df = df[~df['ocr'].str.contains("jboss")]
        df['string_length'] = df['ocr'].str.len()
        df['character_proportion'] = df['string_length'] / df['area']
        print('filtering images')
        df = filter_images(df)
        print(total number of filtered ads {}).format(df.shape[0])
        image_extract(df)

def image_extract(x):
    #x_sample = x.sample(n) # needed if you only want to download a sample
    img_urls = x['image_url']
    id_s = x['id']
    year = x['year'].values[0]
    print(year)
    counter = 0
    for id_, url in zip(id_s, img_urls):
        print(id_)
        id_ = id_.replace(':', '-')
        id_ = 'KBNRC01-' + id_
        print(counter)
        if counter % 100 == 0:
            time.sleep(1)
        else:
            pass
        try:
            link = urlopen(url)
        except Exception:
            print("Bad URL")
            continue
        try:
            name = "{}_{}.jpg".format(year, id_)
            with open(name, "wb") as output:
                output.write(link.read())
        except IOError:
            print("Unable to create %s") % name
        counter += 1


def filter_images(df):
    full_page_ads = (df['area'] > 80000000)
    classifieds = (df['h'] > 5000) & (df['w'] < 900)
    small_ads = (df['h'] < 500) | (df['w'] < 500)

    df = df[df['character_proportion'] < 0.0001]
    #low_character_ads = df[(df['character_proportion'] > 0) & (df['character_proportion'] < 0.0001)]
    df_filter = df[(~full_page_ads) & (~classifieds) & (~small_ads)]
    #df_filter = df[(~classifieds) & (~small_ads) & (~full_page_ads) &(~low_character_ads)]
    return df_filter


if __name__ == '__main__':
    load_file()
