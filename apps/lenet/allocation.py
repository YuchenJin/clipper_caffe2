import itertools

THRESHOLD = 0.03

# Lenet takes 0.2 GPU, and others are 20 googleNets whose rps follow zipf-0.9
apps = [0.2, 4.83375591, 2.59034565, 1.79835566, 1.38813187, 1.13556425, 0.96371494, 0.83887351, 0.74388145, 0.66906214, 0.60853381, 0.55851044, 0.51644205, 0.48054681, 0.44954118, 0.42247653, 0.3986362, 0.37746848, 0.35854153, 0.3415124, 0.3261052]
allocated_apps = set()
allocated_app_pairs = []

for pair in itertools.combinations(apps, r=2):
    if abs(1 - (sum(pair))%1) <= THRESHOLD:
        if not any(i in pair for i in allocated_apps):
            allocated_apps.add(pair[0])
            allocated_apps.add(pair[1])
            allocated_app_pairs.append(pair)
apps = [item for item in apps if item not in allocated_apps]

for pair in itertools.combinations(apps, r=3):
    if abs(1 - (sum(pair))%1) <= THRESHOLD:
        if not any(i in pair for i in allocated_apps):
            allocated_apps.add(pair[0])
            allocated_apps.add(pair[1])
            allocated_apps.add(pair[2])
            allocated_app_pairs.append(pair)
apps = [item for item in apps if item not in allocated_apps]

for i in allocated_app_pairs:
    print i

print ("apps unallocated:" + str(apps))
