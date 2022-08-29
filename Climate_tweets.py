import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "2050 탄소중립 until:2022-08-10 since:2020-10-28"
tweets = []
limit = 25000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    #print(vars(tweet))
    #break

    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.content])

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

df.to_csv('Climate_SouthKorea_tweets_Korean_New.csv')