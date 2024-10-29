import json

# 从文件中读取JSON数据
with open(r'D:\postgraduate0\YHJC\Fire_smoke_monitoring_system-master\config\config.json', 'r') as f:
    data = json.load(f)

print(type(data['iou']))
