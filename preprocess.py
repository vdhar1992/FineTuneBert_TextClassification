import pandas as pd 
import numpy as np 
import regex as re 
from nltk.stem import WordNetLemmatizer


lookup_dict = {
  'abt' : 'about',
  'afaik' : 'as far as i know',
  'bc' : 'because',
  'bfn' : 'bye for now',
  'bgd' : 'background',
  'bh' : 'blockhead',
  'br' : 'best regards',
  'btw' : 'by the way',
  'cc': 'carbon copy',
  'chk' : 'check',
  'dam' : 'do not annoy me',
  'dd' : 'dear daughter',
  'df': 'dear fiance',
  'ds' : 'dear son',
  'dyk' : 'did you know',
  'em': 'email',
  'ema' : 'email address',
  'ftf' : 'face to face',
  'fb' : 'facebook',
  'ff' : 'follow friday',
  'fotd' : 'find of the day',
  'ftw': 'for the win',
  'fwiw' : 'for what it is worth',
  'gts' : 'guess the song',
  'hagn' : 'have a good night',
  'hand' : 'have a nice day',
  'hotd' : 'headline of the day',
  'ht' : 'heard through',
  'hth' : 'hope that helps',
  'ic' : 'i see',
  'icymi' : 'in case you missed it',
  'idk' : 'i do not know',
  'ig': 'instagram',
  'iirc' : 'if i remember correctly',
  'imho' : 'in my humble opinion',
  'imo' : 'in my opinion',
  'irl' : 'in real life',
  'iwsn' : 'i want sex now',
  'jk' : 'just kidding',
  'jsyk' : 'just so you know',
  'jv' : 'joint venture',
  'kk' : 'cool cool',
  'kyso' : 'knock your socks off',
  'lmao' : 'laugh my ass off',
  'lmk' : 'let me know',
  'lo' : 'little one',
  'lol' : 'laugh out loud',
  'mm' : 'music monday',
  'mirl' : 'meet in real life',
  'mrjn' : 'marijuana',
  'nbd' : 'no big deal',
  'nct' : 'nobody cares though',
  'njoy' : 'enjoy',
  'nsfw' : 'not safe for work',
  'nts' : 'note to self',
  'oh' : 'overheard',
  'omg': 'oh my god',
  'oomf' : 'one of my friends',
  'orly' : 'oh really',
  'plmk' : 'please let me know',
  'pnp' : 'party and play',
  'qotd' : 'quote of the day',
  're' : 'in reply to in regards to',
  'rtq' : 'read the question',
  'rt' : 'retweet',
  'sfw' : 'safe for work',
  'smdh' : 'shaking my damn head',
  'smh' : 'shaking my head',
  'so' : 'significant other',
  'srs' : 'serious',
  'tftf' : 'thanks for the follow',
  'tftt' : 'thanks for this tweet',
  'tj' : 'tweetjack',
  'tl' : 'timeline',
  'tldr' : 'too long did not read',
  'tmb' : 'tweet me back',
  'tt' : 'trending topic',
  'ty' : 'thank you',
  'tyia' : 'thank you in advance',
  'tyt' : 'take your time',
  'tyvw' : 'thank you very much',
  'w': 'with',
  'wtv' : 'whatever',
  'ygtr' : 'you got that right',
  'ykwim' : 'you know what i mean',
  'ykyat' : 'you know you are addicted to',
  'ymmv' : 'your mileage may vary',
  'yolo' : 'you only live once',
  'yoyo' : 'you are on your own',
  'yt': 'youtube',
  'yw' : 'you are welcome',
  'zomg' : 'oh my god to the maximum'
}

def abbrev_conversion(text):
    words = text.split()
    abbrevs_removed = []

    for i in words:
        if i in lookup_dict:
            i = lookup_dict[i]
        abbrevs_removed.append(i)

    return ' '.join(abbrevs_removed)


def preprocess_text(text):

    digit_pattern = re.compile(r'[\d]+')

    text = re.sub('[^A-Za-z]+',' ',text)
    text= text.lower()
    text= abbrev_conversion(text)
    text= re.sub(digit_pattern, '', text)

    return text