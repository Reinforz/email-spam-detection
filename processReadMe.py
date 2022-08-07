import os
import re

source = "pd_project.md"
dest = "README.md"

print(f"Reading File {source}")
with open(source) as f:
  mdFile = f.read()

mdFile = re.sub(r"<style scoped>.*</style>", "", mdFile)
print("deleted style scoped tags")

with open(dest, 'w') as f:
  f.write(mdFile)

print(f"Successfully Generated {dest}")

os.remove(source)
print(f"Successfully Deleted {source}")
