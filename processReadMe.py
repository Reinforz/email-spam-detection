from bs4 import BeautifulSoup
import os

source = "pd_project.md"
dest = "README.md"

print(f"Reading File {source}")
with open(source) as f:
  mdFile = f.read()

soup = BeautifulSoup(mdFile, 'html.parser')

styles = soup.find_all('style')
print("deleting style tags")
for styleTag in styles:
  styleTag.decompose()


with open(dest, 'w') as f:
  f.write(soup.prettify())

print(f"Successfully Generated {dest}")

os.remove(source)
print(f"Successfully Deleted {source}")
