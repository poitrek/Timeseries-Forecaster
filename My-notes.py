
'''
Modules learned:

flask-caching - part of the Flask library which is used for caching data from an app.
Can be used to store state between callbacks, eg. models for data prediction.
cache = Cache(config=...)

@cache.memoize()
def get_some_data():
...
http://brunorocha.org/python/flask/using-flask-cache.html
https://dash.plot.ly/sharing-data-between-callbacks

pickle - serialization and deserialization of data in Python. Can convert most of classes,
including custom complex classes. Native for Python

dumps(data) - returns serialized data
loads(serial) - returns deserialized data

https://docs.python.org/3/library/pickle.html#module-pickle

'''