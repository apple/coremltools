import os
import re
import inflection
from itertools import izip

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    l = iter(iterable)
    return izip(l, l)

text_file = open("./_sources/reference.rst","r")
text = text_file.read()

os.remove("./_sources/reference.rst")

if not os.path.exists("./_sources/reference"):
    os.makedirs("./_sources/reference")

sections = map(str.strip, re.split("<!--\s*(.+)\s*-->", text))
for section, content in pairwise(sections[1:]):
    if section.endswith(".proto"):
        file_name = section[:-len(".proto")]
        title = inflection.titleize(file_name)
                
        if title == "Svm":
            title = "SVM"
        elif title == "Glm Classifier":
            title = "GLM Classifier"
        elif title == "Glm Regressor":
            title = "GLM Regressor"
    
        f = open("./_sources/reference/{0}.rst".format(file_name), "w")
        f.truncate()
        f.write(title)
        f.write("\n================================================================================\n")
        f.write("\n")
        f.write(content)
        f.close()
