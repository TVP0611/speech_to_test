import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

with open('bat_den_ap_tran.flac', 'rb') as f:
    flac_data = f.read()
# print('done')
class RequestError(Exception): pass
class UnknownValueError(Exception): pass

language="vi-VN"

key = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
    "client": "chromium",
    "lang": language,
    "key": key,
}))
request = Request(url, data=flac_data, headers={"Content-Type": "audio/x-flac; rate={}".format(8000)})

# obtain audio transcription results
try:
    response = urlopen(request, timeout=None)
except HTTPError as e:
    raise RequestError("recognition request failed: {}".format(e.reason))
except URLError as e:
    raise RequestError("recognition connection failed: {}".format(e.reason))
response_text = response.read().decode("utf-8")

# ignore any blank blocks
actual_result = []
for line in response_text.split("\n"):
    if not line: continue
    result = json.loads(line)["result"]
    if len(result) != 0:
        actual_result = result[0]
        break

# return results
if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: raise UnknownValueError()

if "confidence" in actual_result["alternative"]:
    # return alternative with highest confidence score
    best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
else:
    # when there is no confidence available, we arbitrarily choose the first hypothesis.
    best_hypothesis = actual_result["alternative"][0]
if "transcript" not in best_hypothesis: raise UnknownValueError()
text = best_hypothesis["transcript"]
print(text)