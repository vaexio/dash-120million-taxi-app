
# Benchmarking

The benchmarks are run by disabling the cache
```
$ DASH_CACHE_TIMEOUT=-1 VAEX_NUM_THREADS=8 gunicorn -w 16 app:server -b 0.0.0.0:8050
```


## Light operation
```bash
$ ab  -n 300 -c 64 -H 'Cache-Control: no-cache' -T application/json -p payload_geomap_light.json http://localhost:8050/_dash-update-component
This is ApacheBench, Version 2.3 <$Revision: 1843412 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking localhost (be patient)
Completed 100 requests
Completed 200 requests
Completed 300 requests
Finished 300 requests


Server Software:        gunicorn/20.0.4
Server Hostname:        localhost
Server Port:            8050

Document Path:          /_dash-update-component
Document Length:        247328 bytes

Concurrency Level:      64
Time taken for tests:   7.634 seconds
Complete requests:      300
Failed requests:        0
Total transferred:      74245200 bytes
Total body sent:        68357100
HTML transferred:       74198400 bytes
Requests per second:    39.30 [#/sec] (mean)
Time per request:       1628.536 [ms] (mean)
Time per request:       25.446 [ms] (mean, across all concurrent requests)
Transfer rate:          9497.95 [Kbytes/sec] received
                        8744.70 kb/s sent
                        18242.65 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.8      0       3
Processing:   147 1435 364.3   1576    1678
Waiting:      143 1434 364.4   1576    1678
Total:        147 1435 363.9   1576    1678

Percentage of the requests served within a certain time (ms)
  50%   1576
  66%   1599
  75%   1610
  80%   1619
  90%   1635
  95%   1649
  98%   1659
  99%   1663
 100%   1678 (longest request)
```


## Heavy operation
```bash
$ ab  -n 300 -c 64 -H 'Cache-Control: no-cache' -T application/json -p payload_heatmap_heavy.json http://localhost:8050/_dash-update-component
This is ApacheBench, Version 2.3 <$Revision: 1843412 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking localhost:8050 (be patient)
Completed 100 requests
Completed 200 requests
Completed 300 requests
Finished 300 requests


Server Software:        nginx/1.17.10
Server Hostname:        localhost:8050
Server Port:            80

Document Path:          /_dash-update-component
Document Length:        1619054 bytes

Concurrency Level:      64
Time taken for tests:   27.546 seconds
Complete requests:      300
Failed requests:        0
Total transferred:      485765400 bytes
Total body sent:        234900
HTML transferred:       485716200 bytes
Requests per second:    10.89 [#/sec] (mean)
Time per request:       5876.426 [ms] (mean)
Time per request:       91.819 [ms] (mean, across all concurrent requests)
Transfer rate:          17221.54 [Kbytes/sec] received
                        8.33 kb/s sent
                        17229.87 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        1   14  12.5     12      50
Processing:   361 5252 919.0   5416    7315
Waiting:      288 4888 1025.1   5176    6489
Total:        370 5266 916.9   5428    7329

Percentage of the requests served within a certain time (ms)
  50%   5428
  66%   5596
  75%   5671
  80%   5743
  90%   6023
  95%   6254
  98%   7111
  99%   7281
 100%   7329 (longest request)
```