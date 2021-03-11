
# %% ---------------------------------------------------------------------------

import credentials
import tweepy

# %% ---------------------------------------------------------------------------
# put the authorization codes here from your twitter developer application
# CONSUMER_KEY = 'Bukz3iJOZ5VQRpkEOXgZNdUvR'   # 'PAx7RXiuo6JTjikKZH94hKliI'
# CONSUMER_SECRET = 'YhGmJT2JtXs58ci3ymN1zc9XLtJOUhmqsIpnDTGym8qsIZgEOn'  #'toV6d2hAKNvGruzvpZUcrKkCphv5gMwHWx4ip788jTvJv6WZyW'
# OAUTH_TOKEN = '14167183-qOYJ1Q2o0ERoBgVPYXu2CZFjXkwBUq1UJMqfSl4Vu'  # '4011915507-1qo7Y5Bn216BKoOsMc6nOya4yEewMqSyyhxCNee'
# OAUTH_SECRET = 'Ezfr6bnYpOVmFj6MwNoP9goqzJezzJxsJqLecVlv552MG' # 'Hgx7NBblGxBOGlsL133IObUWbmj4AmHCEizjPzn9BejTD'
          
# %% ---------------------------------------------------------------------------
# login to Twitter with ordinary rate limiting
def oauth_login():
    # get the authorization from Twitter and save in the twepy package
    auth = tweepy.OAuthHandler(
        credentials.TWITTER_CONSUMER_KEY,
        credentials.TWITTER_CONSUMER_SECRET)
    auth.set_access_token(
        credentials.TWITTER_OAUTH_TOKEN,
        credentials.TWITTER_OAUTH_SECRET)
    api = tweepy.API(auth)

    # Ensure an API is returned
    assert api

    # Return the API object.
    return api
# %% ---------------------------------------------------------------------------
# login to Twitter with extended rate limiting
# must be used with the tweepy Cursor to wrap the search and enact the waits
def appauth_login():
    # get the authorization from Twitter and save in the twepy package
    auth = tweepy.AppAuthHandler(
        credentials.TWITTER_CONSUMER_KEY,
        credentials.TWITTER_CONSUMER_SECRET)
    # apparently no need to set the other access tokens
    api = tweepy.API(
        auth, 
        wait_on_rate_limit = True, 
        wait_on_rate_limit_notify = True)

    # Ensure an API is returned
    assert api

    # Return the API object
    return api
# %% ---------------------------------------------------------------------------

# Test program to show how to connect
tapi = oauth_login()
print ("Twitter OAuthorization: ", tapi)
# tapi = appauth_login()
# print ("Twitter AppAuthorization: ", tapi)

# %% ---------------------------------------------------------------------------

out = tapi.user_timeline(screen_name='lta100163', count=10)
type(out)

# %% ---------------------------------------------------------------------------

def grab_prior_days_tweets (qstr) :
    tweets = tapi.search(
        q='"The_GIJoe_Club"',
        count=100,
        lang="en",
        result_type = "recent",
        max_id = 1368385423303606274,
        tweet_mode='extended')
    for tweet in tweets :
        print("-" * 80)
        print(tweet.full_text)
        print(f"\tID: {tweet.id}")
        print(f"\tCREATED AT: {tweet.created_at}")
        print(f"\tHASHTAGS: {[ht.get('text') for ht in tweet.entities.get('hashtags')]}")
        recount_level = "LOW" if tweet.retweet_count < 10 else "HIGH" if tweet.retweet_count > 100 else ""
        print(f"\tRETWEET COUNT: {tweet.retweet_count} {recount_level}")
        print("\n")


# %% ---------------------------------------------------------------------------

#tweets = tweepy.Cursor(tweepy_api.search, q="rstats", lang="en").items(20)

import datetime as dt 
yesterday = (dt.datetime.now() - dt.timedelta(days=1)).date()

more_tweets = True
last_id = 99999999999999999999

while more_tweets :
    tweets = tapi.search(
        q='Hasbro',
        count=100,
        lang="en",
        result_type = "recent",
        max_id = last_id,
        # until=dt.date.today().strftime('%Y-%m-%d'),
        tweet_mode='extended')
    if len(tweets) > 0 :
        last_id = tweets[len(tweets)-1].id
    if len(tweets) < 100 :
        more_tweets = False
    print(f'{len(tweets)} - {last_id}')


# for tweet in tweets :
#     print("-" * 80)
#     print(tweet.full_text)
#     print(f"\tID: {tweet.id}")
#     print(f"\tCREATED AT: {tweet.created_at} ({type(tweet.created_at)})")
#     print(f"\tHASHTAGS: {[ht.get('text') for ht in tweet.entities.get('hashtags')]}")
#     recount_level = "LOW" if tweet.retweet_count < 10 else "HIGH" if tweet.retweet_count > 100 else ""
#     print(f"\tRETWEET COUNT: {tweet.retweet_count} {recount_level}")
#     print("\n")

# %%----------------------------------------------------

import pandas as pd

# Use a very large number up front so we get the most recent tweet.
last_id = 99999999999999999999

# Initialize total tweet counter
total_tweets = 0

# Initialize a dataframe for the result
df = pd.DataFrame()

# Loop until no more tweets are available.
more_tweets = True
while more_tweets :
    tweets = tapi.search(			# Search for tweets
        q="hottoysofficial",						# Use the given query string
        count=100,					# Get up to 100 (the max) per request
        lang="en",					# Look for only English tweets
        result_type = "recent",		# Order from most to least recent
        max_id = last_id,			# The starting point for a new batch
        tweet_mode='extended')		# Get full text
    
    # Update the tweet counts for the batch and the total
    num_tweets_in_batch = len(tweets)
    total_tweets += num_tweets_in_batch

    # Add the tweets to the dataframe
    new_tweet_list = [t._json for t in tweets]
    df=df.append(new_tweet_list, ignore_index=True)

    if num_tweets_in_batch > 0 :
        # If any tweets were returned use the ID of the last tweet as the
        # starting point for the next batch.
        last_id = tweets[num_tweets_in_batch-1].id
    if num_tweets_in_batch < 100 or total_tweets >= 1000 :
        # If we recieved less than the requested number of tweets, that's 
        # all there is. No more tweets.
        more_tweets = False
    print(f'{len(tweets)} - {last_id}')

print(df)
# %%
