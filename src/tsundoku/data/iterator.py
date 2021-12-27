import sys

import rapidjson as json

from tsundoku.features.tweets import flatten_tweet


def iterate_tweets(src, encoding="utf-8"):
    seen_tweets = set()

    for l in src:
        try:
            line = l.decode(encoding)  # .encode('utf-8')
            # print(line)
            # raise Exception()
            if not line.startswith("{"):
                continue

            tweet = json.loads(line)

            if type(tweet) == dict and "user" in tweet:
                if tweet["id"] in seen_tweets:
                    continue

                seen_tweets.add(tweet["id"])

                if "retweeted_status" in tweet:
                    rt = tweet["retweeted_status"]
                    if not rt["id"] in seen_tweets:
                        seen_tweets.add(rt["id"])
                        yield flatten_tweet(rt)

                if "quoted_status" in tweet:
                    quote = tweet["quoted_status"]

                    if not quote["id"] in seen_tweets:
                        seen_tweets.add(quote["id"])
                        yield flatten_tweet(quote)

                yield flatten_tweet(tweet)

        except UnicodeDecodeError as ex:
            print("UnicodeError:", l, file=sys.stderr)
            continue
        except ValueError as ex:
            print("ValueError:", l, file=sys.stderr)
            continue
