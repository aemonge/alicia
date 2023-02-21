import sys, json

with open(sys.argv[1], 'r') as f:
  categories = json.load(f)
fixed_categories = { (int(key) - 1): val for key, val in categories.items() }

# save fixed_categories into file
with open(sys.argv[1], 'w') as f:
  json.dump(fixed_categories, f)
