import os
import openai
openai.api_key = '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'
a = openai.Model.list()

f = open("./models.txt", "w")
f.write(str(a))
