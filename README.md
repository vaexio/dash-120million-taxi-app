# dash-117million-taxi-app
Explore 117 million taxi trips in real time with Dash and Vaex


# Running this app

Clone the repo
```
$ git clone https://github.com/vaexio/dash-117million-taxi-app
```

Run in debug mode:
```
$ python app.py
```

Run in production mode:
```
$ VAEX_NUM_THREADS=8 gunicorn -w 16 app:server -b 0.0.0.0:8050
```

