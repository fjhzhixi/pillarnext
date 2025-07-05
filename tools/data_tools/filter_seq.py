import json

with open('tools/data_tools/output/metrics_0.3.json', 'r') as f:
    data = json.load(f)

data = dict(sorted(data.items(), key=lambda x: x[1]['mAP'], reverse=False))

res = []
cnt = 0
for key, value in data.items():
    res.append(key)
    # if value['mAP'] < 2:
    #     cnt += 1
        # print(key, value['mAP'])

# print(cnt)
print(res[:40])

# import ipdb; ipdb.set_trace()
